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

is_command() { hash "$1" 2>/dev/null; }
apt_is_locked() { fuser /var/lib/dpkg/lock >/dev/null 2>&1; }
wait_for_apt() {
    if apt_is_locked; then
        echo "Waiting to obtain dpkg lock file..."
        while apt_is_locked; do echo .; sleep 0.5; done
    fi
}
vpython() { "$VENV/bin/python" $@; }
vpip() { "$VENV/bin/pip" $@; }

#############################################
set -e; cd "$(dirname "$0")" # Script Start #
#############################################

VENV=${VENV-$(pwd)/.venv}

if is_command apt-get; then
	wait_for_apt
	sudo apt-get install -y python3-pip libopenblas-dev python3-scipy cython libhdf5-dev python3-h5py portaudio19-dev
fi

if [ ! -x "$VENV/bin/pip" ]; then
    python3 -m venv "$VENV" --without-pip
    curl https://bootstrap.pypa.io/get-pip.py | vpython
fi

arch="$(python3 -c 'import platform; print(platform.machine())')"

if ! vpython -c 'import tensorflow' 2>/dev/null && [ "$arch" = "armv7l" ]; then
    whl=tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl
    wget "https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v1.1.0/$whl"
	vpip install "$whl"
	vpip uninstall mock || true; vpip install mock
    rm "$whl"
fi

vpip install -e runner/
vpip install -e .
vpip install pocketsphinx  # Optional, for comparison
