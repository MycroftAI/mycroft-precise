#!/usr/bin/env bash

if ! [[ "$1" =~ .*\.net$ ]]; then
    echo "Usage: $0 <model>.net"
    exit 1
fi

[ -d .cache/precise-data ] || git clone https://github.com/mycroftai/precise-data .cache/precise-data
model_name=$(date +"${1%%net}%y-%m-%d")
precise/scripts/convert.py $1 -o ".cache/precise-data/$model_name.pb"
cp "$1" ".cache/precise-data/$model_name.net"
cp "$1.params" ".cache/precise-data/$model_name.net.params"
mv "$model_name.pb" "$model_name.pb.params" ".cache/precise-data/"
echo "Converted to .cache/precise-data/$model_name.*"
