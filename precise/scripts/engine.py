#!/usr/bin/env python3
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
"""
stdin should be a stream of raw int16 audio, written in
groups of CHUNK_SIZE samples. If no CHUNK_SIZE is given
it will read until EOF. For every chunk, an inference
will be given via stdout as a float string, one per line

:model_name str
    Keras or TensorFlow model to read from

...
"""
import sys

import os
from prettyparse import Usage

from precise import __version__
from precise.network_runner import Listener
from precise.scripts.base_script import BaseScript


def add_audio_pipe_to_parser(parser):
    parser.usage = parser.format_usage().strip().replace('usage: ', '') + ' < audio.wav'


class EngineScript(BaseScript):
    usage = Usage(__doc__)
    usage.add_argument('-v', '--version', action='version', version=__version__)
    usage.add_argument('chunk_size', type=int, nargs='?', default=-1,
                       help='Number of bytes to read before making a prediction. '
                            'Higher values are less computationally expensive')
    usage.add_customizer(add_audio_pipe_to_parser)

    def __init__(self, args):
        super().__init__(args)
        if sys.stdin.isatty():
            raise ValueError('Please pipe audio via stdin using < audio.wav')

    def run(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        stdout = sys.stdout
        sys.stdout = sys.stderr
        listener = Listener(self.args.model_name, self.args.chunk_size)

        try:
            while True:
                conf = listener.update(sys.stdin.buffer)
                stdout.buffer.write((str(conf) + '\n').encode('ascii'))
                stdout.buffer.flush()
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            sys.stdout = stdout


main = EngineScript.run_main

if __name__ == '__main__':
    main()
