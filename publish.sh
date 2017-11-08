#!/usr/bin/env bash

# Usage: upload_file FILE REMOTE_PATH
upload_file() {
    file="$1"
    remote_url="s3://$2"
	eval cfg_file="~/.s3cfg.mycroft-artifact-writer"
    [ -f "$cfg_file" ] && s3cmd put $1 $remote_url --acl-public -c ~/.s3cfg.mycroft-artifact-writer || echo "Could not find $cfg_file. Skipping upload."
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

# Usage: check_args "$@"
check_args() {
    if [ $# != 1 ]; then
        echo "Usage: $1 stable|unstable"
        exit 1
    fi
}

set -e

check_args "$@"

type="$(find_type $@)"
version="$(find_version $@)"
arch="$(find_arch)"

sudo pip3 install pyinstaller
pyinstaller -y precise.stream.spec

echo $version > latest
upload_file dist/precise-stream bootstrap.mycroft.ai/artifacts/static/$type/$arch/$version/
upload_file latest bootstrap.mycroft.ai/artifacts/static/$type/$arch/

