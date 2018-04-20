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

if ! [[ "$1" =~ .*\.net$ ]] || ! [ -d "$2" ]; then
	echo "Usage: $0 <model>.net <folder_name>"
	exit 1
fi

[ -d .cache/precise-data ] || git clone https://github.com/mycroftai/precise-data .cache/precise-data
model_name=$(date +"${1%%net}%y-%m-%d")
precise/scripts/convert.py $1 -o "$2/$model_name.pb"
cp "$1" "$2/$model_name.net"
cp "$1.params" "$2/$model_name.net.params"
mv "$model_name.pb" "$model_name.pb.params" "$2"
echo "Converted to $2/$model_name.*"
