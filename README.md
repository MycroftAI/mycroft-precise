# Mycroft Precise

*A lightweight, simple-to-use, RNN wake word listener.*

Precise is a wake word listener. Like its name suggests,
a wake word listener's job is to continually listen to
sounds and speech around the device, and activate when
the sounds or speech match a wake word. Unlike other machine
learning hotword detection tools, Mycroft Precise is fully open
source. Take a look at a [comparison here][comparison].

[comparison]: https://github.com/MycroftAI/mycroft-precise/wiki/Software-Comparison

## Training Models

### Communal models

Training takes lots of data.  The Mycroft community is working together to jointly
build datasets at https://precise.mycroft.ai.  These datasets are used to build the
models used by the Mark 1 and other mycroft-core based voice assistants.  Please come
and help make things better for everyone!

### Train your own model

You can find info on training your own models [here][train-guide]. It requires
running through the [**Source Install** instructions][source-install] first.

[train-guide]:https://github.com/MycroftAI/mycroft-precise/wiki/Training-your-own-wake-word#how-to-train-your-own-wake-word
[source-install]:https://github.com/MycroftAI/mycroft-precise#source-install

## Installation

If you just want to use Mycroft Precise for running models in your own application,
you can use the binary install option. If you want to train your own models or mess
with the source code, you'll need to follow the **Source Install** instructions below.

### Binary Install

First download `precise-engine.tar.gz` from the [precise-data][precise-data] GitHub
repo. Currently, we support both 64 bit desktops (x86_64) and the Raspberry Pi (armv7l).

[precise-data]:https://github.com/mycroftai/precise-data/tree/dist

Next, extract the tar to the folder of your choice. The following commands will work for the pi:

```bash
ARCH=armv7l
wget https://github.com/MycroftAI/precise-data/raw/dist/$ARCH/precise-engine.tar.gz
tar xvf precise-engine.tar.gz
```

Now, the Precise binary exists at `precise-engine/precise-engine`.

Next, install the Python wrapper with `pip3` (or `pip` if you are on Python 2):

```bash
sudo pip3 install precise-runner
```

Finally, you can write your program, passing the location of the precise binary like shown:

```python
#!/usr/bin/env python3

from precise_runner import PreciseEngine, PreciseRunner

engine = PreciseEngine('precise-engine/precise-engine', 'my_model_file.pb')
runner = PreciseRunner(engine, on_activation=lambda: print('hello'))
```

### Source Install

Start out by cloning the repository:

```bash
git clone https://github.com/mycroftai/mycroft-precise
cd mycroft-precise
```

Next, install the necessary system dependencies. If you are on Ubuntu, this
will be done automatically in the next step. Otherwise, feel free to submit
a PR to support other operating systems. The dependencies are:

 - python3-pip
 - libopenblas-dev
 - python3-scipy
 - cython
 - libhdf5-dev
 - python3-h5py
 - portaudio19-dev

After this, run the setup script:

```bash
./setup.sh
```

Finally, you can write your program as follows:

```python
#!/usr/bin/env python3

from precise_runner import PreciseEngine, PreciseRunner

engine = PreciseEngine('.venv/bin/precise-engine', 'my_model_file.pb')
runner = PreciseRunner(engine, on_activation=lambda: print('hello'))
```

In addition to the `precise-engine` executable, doing a **Source Install** gives you
access to some other scripts. You can read more about them [here][executables].
One of these executables, `precise-listen`, can be used to test a model using
your microphone:

[executables]:https://github.com/MycroftAI/mycroft-precise/wiki/Training-your-own-wake-word#how-to-train-your-own-wake-word

```bash
source .venv/bin/activate  # Gain access to precise-* executables
precise-listen my_model_file.pb
```

## How it Works

At it's core, Precise uses just a single recurrent network, specifically a GRU.
Everything else is just a matter of getting data into the right form.

![Architecture Diagram](https://images2.imgbox.com/f7/44/6N4xFU7D_o.png)
