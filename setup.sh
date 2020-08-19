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

is_command() { hash "$1" 2>/dev/null; }
apt_is_locked() { fuser /var/lib/dpkg/lock >/dev/null 2>&1; }
wait_for_apt() {
	if apt_is_locked; then
		echo "Waiting to obtain dpkg lock file..."
		while apt_is_locked; do echo .; sleep 0.5; done
	fi
}
has_piwheels() { cat /etc/pip.conf 2>/dev/null | grep -qF 'piwheels'; }
install_piwheels() {
    echo "Installing piwheels..."
    echo "
[global]
extra-index-url=https://www.piwheels.org/simple
" | sudo tee -a /etc/pip.conf
}

#############################################
set -e; cd "$(dirname "$0")" # Script Start #
#############################################

VENV=${VENV-$(pwd)/.venv}

os=$(uname -s)
if [ "$os" = "Linux" ]; then
    if is_command apt-get; then
        wait_for_apt
        sudo apt-get install -y python3-pip curl libopenblas-dev python3-scipy cython libhdf5-dev python3-h5py portaudio19-dev swig libpulse-dev libatlas-base-dev
    fi
elif [ "$os" = "Darwin" ]; then
    if is_command brew; then
        brew install portaudio
    fi
fi

if [ ! -x "$VENV/bin/python" ]; then python3 -m venv "$VENV" --without-pip; fi
source "$VENV/bin/activate"
if [ ! -x "$VENV/bin/pip" ]; then curl https://bootstrap.pypa.io/get-pip.py | python; fi

arch="$(python -c 'import platform; print(platform.machine())')"

if [ "$arch" = "armv7l" ] && ! has_piwheels; then
    install_piwheels
fi

pip install -e runner/
pip install -e .
pip install pocketsphinx  # Optional, for comparison
