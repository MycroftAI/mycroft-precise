#!/usr/bin/env bash

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

wait_for_apt
sudo apt-get install -y python3-pip libblas-dev python3-scipy cython python3-h5py portaudio19-dev

git fetch && git reset --hard origin/rnn

arch="$(python3 -c 'import platform; print(platform.machine())')"

if ! python3 -c 'import tensorflow' 2>/dev/null && [ "$arch" = "armv7l" ]; then
    wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v1.1.0/tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl
	sudo pip3 install tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl
	sudo pip3 uninstall mock || true
    sudo pip3 install mock
fi

sudo pip3 install --upgrade pip
sudo pip3 install -e .

