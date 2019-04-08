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
import numpy as np
from os.path import join
from prettyparse import create_parser
from random import randint
from shutil import get_terminal_size
from threading import Event

from precise.network_runner import Listener
from precise.util import save_audio, buffer_to_audio, activate_notify
from precise_runner import PreciseRunner
from precise_runner.runner import ListenerEngine

usage = '''
    Run a model on microphone audio input
    
    :model str
        Either Keras (.net) or TensorFlow (.pb) model to run
    
    :-c --chunk-size int 2048
        Samples between inferences
    
    :-l --trigger-level int 3
        Number of activated chunks to cause an activation
    
    :-s --sensitivity float 0.5
        Network output required to be considered activated

    :-b --basic-mode
        Report using . or ! rather than a visual representation

    :-d --save-dir str -
        Folder to save false positives
    
    :-p --save-prefix str -
        Prefix for saved filenames
'''

session_id, chunk_num = '%09d' % randint(0, 999999999), 0


def main():
    args = create_parser(usage).parse_args()

    def on_activation():
        activate_notify()

        if args.save_dir:
            global chunk_num
            nm = join(args.save_dir, args.save_prefix + session_id + '.' + str(chunk_num) + '.wav')
            save_audio(nm, audio_buffer)
            print()
            print('Saved to ' + nm + '.')
            chunk_num += 1

    def on_prediction(conf):
        if args.basic_mode:
            print('!' if conf > 0.7 else '.', end='', flush=True)
        else:
            max_width = 80
            width = min(get_terminal_size()[0], max_width)
            units = int(round(conf * width))
            bar = 'X' * units + '-' * (width - units)
            cutoff = round((1.0 - args.sensitivity) * width)
            print(bar[:cutoff] + bar[cutoff:].replace('X', 'x'))

    listener = Listener(args.model, args.chunk_size)
    audio_buffer = np.zeros(listener.pr.buffer_samples, dtype=float)

    def get_prediction(chunk):
        nonlocal audio_buffer
        audio = buffer_to_audio(chunk)
        audio_buffer = np.concatenate((audio_buffer[len(audio):], audio))
        return listener.update(chunk)

    engine = ListenerEngine(listener, args.chunk_size)
    engine.get_prediction = get_prediction
    runner = PreciseRunner(engine, args.trigger_level, sensitivity=args.sensitivity,
                           on_activation=on_activation, on_prediction=on_prediction)
    runner.start()
    Event().wait()  # Wait forever


if __name__ == '__main__':
    main()
