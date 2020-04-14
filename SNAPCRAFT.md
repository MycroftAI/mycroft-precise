# Mycroft-Precise Snap package

## Package details
The package is a fairly standard configuration using the Snap [Python Plugin](https://snapcraft.io/docs/python-plugin).

The package version is extracted from the `setup.py` script to remain consistent with the pip information. In addition a comparison with the repo is made to see if this commit matches the latest release commit. If it doesn't match _-dev_ is appended to the version info.

Since Precise uses pyaudio, ALSA is required for the audio to work. This Snap uses the [alsa-mixin](https://snapcraft-alsa.readthedocs.io/en/latest/snapcraft_usage.html) to setup the ALSA components. This results in some minor issues that need to be handled. Library conflicts may occur if audio packages are included as part of the main application part, making them appear twice in conflicting versions. This is the reason that libportaudio2 and pulseaudio are in the stage-packages for the alsa-mixin part.

This package also includes the current pretrained "hey mycroft" model accessed through `/snap/mycroft-precise/current/hey-mycroft/hey-mycroft.pb`.

## Plugs
Plugs allow the Snap to connect to specific aspects of the host system. Without any specified plugs the application will run entirely within a read-only file system dedicated to the application. It will not be able to access the system, including audio devices, that are outside of the Snap container.

**Plugs used:**
- home: access to the user's home directory (excluding hidden files)
- audio-record: Access to the system's audio input devices
- audio-playback: Access to the system's audio output devices
