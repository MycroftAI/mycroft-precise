#!/usr/bin/env bash

is_command() {
    hash "$1" 2>/dev/null
}

apt_is_locked() {
    fuser /var/lib/dpkg/lock >/dev/null 2>&1
}

wait_for_apt() {
    if apt_is_locked; then
        echo "Waiting to obtain dpkg lock file..."
        while apt_is_locked; do echo .; sleep 0.5; done
    fi
}

set -e

if is_command apt-get; then
	wait_for_apt
	sudo apt-get install -y python3-pip libopenblas-dev python3-scipy cython libhdf5-dev python3-h5py portaudio19-dev
fi

python=.venv/bin/python
pip=.venv/bin/pip

if [ ! -f "$pip" ]; then
    python3 -m venv .venv/ --without-pip
    curl https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
fi

arch="$(python3 -c 'import platform; print(platform.machine())')"

if ! $python -c 'import tensorflow' 2>/dev/null && [ "$arch" = "armv7l" ]; then
    wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v1.1.0/tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl
	$pip install tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl
	$pip uninstall mock || true
    $pip install mock
    rm tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl
fi

$pip install -e runner/
$pip install -e .

