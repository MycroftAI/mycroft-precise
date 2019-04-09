#!/usr/bin/env bash

set -eE

./setup.sh
source .venv/bin/activate
pip install pyinstaller

rm -rf dist/

for script in listen train calc_threshold; do
	tmp_name=$(mktemp).spec
    cat "precise.template.spec" | sed -e 's/%%SCRIPT%%/'"$script"'/gm' > "$tmp_name"
	pyinstaller -y "$tmp_name"
done

items=dist/*
for i in $items; do
    mkdir -p dist/precise
	cp -R $i/* dist/precise
done

out_name="precise_$(precise-engine --version 2>&1)_$(uname -m).tar.gz"
cd dist
tar czvf "$out_name" "precise"
echo "Wrote to dist/$out_name"
