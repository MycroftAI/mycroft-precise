#!/usr/bin/env bash

set -eE

./setup.sh
source .venv/bin/activate
pip install pyinstaller
pyinstaller -y precise.engine.spec

out_name="dist/precise-engine_$(precise-engine --version 2>&1)_$(uname -m).tar.gz"
tar czvf "$out_name" "dist/precise-engine"
echo "Wrote to $out_name"

