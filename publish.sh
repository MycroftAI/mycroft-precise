#!/usr/bin/env bash

# Usage: upload_file FILE REMOTE_PATH
upload_s3() {
	file="$1"
	remote_url="s3://$2"
	eval cfg_file="~/.s3cfg.mycroft-artifact-writer"
	[ -f "$cfg_file" ] && s3cmd put $1 $remote_url --acl-public -c ~/.s3cfg.mycroft-artifact-writer || echo "Could not find $cfg_file. Skipping upload."
}

# Usage: upload_git FILE GIT_FOLDER
upload_git() {
	[ -d 'precise-data' ] || git clone git@github.com:MycroftAI/precise-data.git
	cd precise-data
	git fetch
	git checkout origin/dist
	mv ../$1 $2
	git add $2
	git commit --amend --no-edit
	git push --force origin HEAD:dist
	cd ..
}

# Usage: find_type stable|unstable
find_type() {
	[ "$1" = "stable" ] && echo "release" || echo "daily"
}

# Usage: find_version stable|unstable
find_version() {
	[ "$1" = "stable" ] && git describe --abbrev=0 || date +%s
}

find_arch() {
	python3 -c 'import platform; print(platform.machine())'
}

# Usage: show_usage $0
show_usage() {
	echo "Usage: $1 stable|unstable [git|s3]"
	exit 1
}

# Usage: parse_args "$@"
parse_args() {
	build_type="error"
	upload_type="s3"
	
	while [ $# -gt 0 ]; do
		case "$1" in
			stable|unstable)
				build_type="$1";;
			git|s3)
				upload_type="$1";;
			*)
				show_usage
		esac
		shift
	done
	[ "$build_type" != "error" ] || show_usage
}

set -e

parse_args "$@"

type="$(find_type $build_type)"
version="$(find_version $build_type)"
arch="$(find_arch)"

sudo pip3 install pyinstaller
pyinstaller -y precise.stream.spec

echo $version > latest

if [ "$upload_type" = "git" ]; then
	upload_git dist/precise-stream $arch/
else
	upload_s3 dist/precise-stream bootstrap.mycroft.ai/artifacts/static/$type/$arch/$version/
	upload_s3 dist/precise-stream bootstrap.mycroft.ai/artifacts/static/$type/$arch/  # Replace latest version
	upload_s3 latest bootstrap.mycroft.ai/artifacts/static/$type/$arch/
fi

