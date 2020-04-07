# Mycroft-Precise Snap package

## Package details

The package is farily standard package using the Snap [Python Plugin](https://snapcraft.io/docs/python-plugin).

The package version is extracted from the setup.py script to remain consistent with the pip information. In addition a comparison with the repo is made to see if this is commit matches the latest release commit, if it doesn't match _-dev_ is applied to the version info.

Since Precise uses pyaudio, ALSA is required for things to work. This requires the [alsa-mixin](https://snapcraft-alsa.readthedocs.io/en/latest/snapcraft_usage.html) and results in some minor issues that needs to be handled. Library conflicts may occur if audio packages are included as part of the main application part, making them appear twice in conflicting versions. This is the reason that libportaudio2 and pulseaudio are in the stage-packages for the alsa-mixin part.

## Plugs
Plugs allows connecting the snap to the rest of the system, without any specified plugs the application will run without being able to access the system outside of the snap container, a read-only file system dedicated to the application.

**Plugs used:**
- home: access to the user's home directory (excluding hidden files)
- audio-record: Access to the system's audio input devices
- audio-playback: Access to the system's audio output devices

## Entry points
Below are some of the entry-points of interest.

- mycroft-precise
Alias for the mycroft-precise.engine, run precise against a data stream over stdio.

- mycroft-precise.listen
Run a model on microphone audio input.

- mycroft-precise.train
Train a new model on a dataset.
