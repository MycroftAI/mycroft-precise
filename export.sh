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

if ! [[ "$1" =~ .*\.net$ ]] || ! [ -f "$1" ] || ! [[ $# =~ [2-3] ]]; then
	echo "Usage: $0 <model>.net GITHUB_REPO [BRANCH]"
	exit 1
fi

model_file=$(readlink -f "$1")
repo=$2
branch=${3-master}

cd "$(dirname "$0")"
set -e
cache=.cache/precise-data/${repo//\//.}.${branch//\//.}
[ -d "$cache" ] || git clone "$repo" "$cache" -b "$branch" --single-branch

pushd "$cache"
git fetch
git checkout "$branch"
git reset --hard "origin/$branch"
popd

source .venv/bin/activate
model_name=$(basename "${1%%.net}")
precise-convert "$model_file" -o "$cache/$model_name.pb"

pushd "$cache"
tar cvf "$model_name.tar.gz" "$model_name.pb" "$model_name.pb.params"
md5sum "$model_name.tar.gz" > "$model_name.tar.gz.md5"
rm -f "$model_name.pb" "$model_name.pb.params" "$model_name.pbtxt"
git reset
git add "$model_name.tar.gz" "$model_name.tar.gz.md5"

echo
ls
git status

read -p "Uploading $model_name model to branch $branch on repo $repo. Confirm? (y/N) " answer
if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
    echo "Aborted."
    exit 1
fi

git commit -m "Update $model_name"
git push
popd

