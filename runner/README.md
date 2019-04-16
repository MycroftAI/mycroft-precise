# Precise Wrapper

A simple to use, lightweight Python module for using Mycroft Precise.

## Usage:

First, download the precise binary from [the precise-data repo][precise-data].
Next, extract the tar to the folder of your choice. The following commands will
work for a desktop:

[precise-data]: https://github.com/mycroftai/precise-data/tree/dist

```bash
ARCH=x86_64
VERSION=0.3.0
wget https://github.com/MycroftAI/mycroft-precise/releases/download/$VERSION/precise-all_${VERSION}_${ARCH}.tar.gz
tar xvf precise-engine.tar.gz
```

Finally, you can create a program as follows, passing in the location of
the executable as the first argument:

```python
#!/usr/bin/env python3

from precise_runner import PreciseEngine, PreciseRunner

engine = PreciseEngine('precise-engine/precise-engine', 'my_model_file.pb')
runner = PreciseRunner(engine, on_activation=lambda: print('hello'))
```
