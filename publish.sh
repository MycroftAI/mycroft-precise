#!/usr/bin/env bash
# Copyright 2019 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

.venv/bin/pip3 install pyinstaller
rm -rf dist/
echo "Building executable..."
.venv/bin/pyinstaller -y precise.engine.spec

out_file=dist/precise-engine.tar.gz
cd dist
tar -czvf "precise-engine.tar.gz" precise-engine
cd -

echo $version > latest

if [ "$upload_type" = "git" ]; then
	upload_git "$out_file" $arch/
else
	upload_s3 "$out_file" bootstrap.mycroft.ai/artifacts/static/$type/$arch/$version/
	upload_s3 "$out_file" bootstrap.mycroft.ai/artifacts/static/$type/$arch/  # Replace latest version
	upload_s3 latest bootstrap.mycroft.ai/artifacts/static/$type/$arch/
fi

