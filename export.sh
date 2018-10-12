#!/usr/bin/env bash
# Copyright 2018 Mycroft AI Inc.
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

if ! [[ "$1" =~ .*\.net$ ]] || ! [ -f "$1" ] || [ "$#" != "1" ]; then
	echo "Usage: $0 <model>.net"
	exit 1
fi

model_file=$(readlink -f "$1")
cd "$(dirname "$0")"
set -e
cache=.cache/precise-data
[ -d "$cache" ] || git clone https://github.com/mycroftai/precise-data "$cache" -b models --single-branch

pushd "$cache"
git fetch
git checkout models
git reset --hard origin/models
popd

source .venv/bin/activate
model_name=$(basename "${1%%.net}")
precise-convert "$model_file" -o "$cache/$model_name.pb"
read -p "Uploading $model_file to public repo with name \"$model_name\". Confirm? (y/n) " response
if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
    return 1
fi

pushd "$cache"
tar cvf "$model_name.tar.gz" "$model_name.pb" "$model_name.pb.params"
md5sum "$model_name.tar.gz" > "$model_name.tar.gz.md5"
rm -f "$model_name.pb" "$model_name.pb.params" "$model_name.pbtxt"
git reset
git add "$model_name.tar.gz" "$model_name.tar.gz.md5"
git commit -m "Update $model_name"
git push
popd

