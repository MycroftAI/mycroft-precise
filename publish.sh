#!/usr/bin/env bash

set -e

sudo pip3 install pyinstaller
pyinstaller -y precise.stream.spec

cd dist/
[ -d repo ] || git clone https://github.com/MycroftAI/precise-data -b dist repo
cd repo/

arch="$(python3 -c 'import platform; print(platform.machine())')"
mv ../precise-stream "$arch/"

if ! git config user.name || ! git config user.email; then
    read -p "Enter git username:" name
    read -p "Enter git email:" email
    git config credential.helper store
    git config --global user.name "$name"
    git config --global user.email "$email"
fi

git commit --amend --no-edit
git push origin dist --force
cd ../../

