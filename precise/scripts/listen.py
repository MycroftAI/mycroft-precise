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
"""
import numpy as np
from os.path import join
from precise_runner import PreciseRunner
from precise_runner.runner import ListenerEngine
from prettyparse import Usage
from random import randint
from shutil import get_terminal_size
from threading import Event

from precise.network_runner import Listener
from precise.scripts.base_script import BaseScript
from precise.util import save_audio, buffer_to_audio, activate_notify


class ListenScript(BaseScript):
    usage = Usage(__doc__)

    def __init__(self, args):
        super().__init__(args)
        self.listener = Listener(args.model, args.chunk_size)
        self.audio_buffer = np.zeros(self.listener.pr.buffer_samples, dtype=float)
        self.engine = ListenerEngine(self.listener, args.chunk_size)
        self.engine.get_prediction = self.get_prediction
        self.runner = PreciseRunner(self.engine, args.trigger_level, sensitivity=args.sensitivity,
                                    on_activation=self.on_activation, on_prediction=self.on_prediction)
        self.session_id, self.chunk_num = '%09d' % randint(0, 999999999), 0

    def on_activation(self):
        activate_notify()

        if self.args.save_dir:
            nm = join(self.args.save_dir, self.args.save_prefix + self.session_id + '.' + str(self.chunk_num) + '.wav')
            save_audio(nm, self.audio_buffer)
            print()
            print('Saved to ' + nm + '.')
            self.chunk_num += 1

    def on_prediction(self, conf):
        if self.args.basic_mode:
            print('!' if conf > 0.7 else '.', end='', flush=True)
        else:
            max_width = 80
            width = min(get_terminal_size()[0], max_width)
            units = int(round(conf * width))
            bar = 'X' * units + '-' * (width - units)
            cutoff = round((1.0 - self.args.sensitivity) * width)
            print(bar[:cutoff] + bar[cutoff:].replace('X', 'x'))

    def get_prediction(self, chunk):
        audio = buffer_to_audio(chunk)
        self.audio_buffer = np.concatenate((self.audio_buffer[len(audio):], audio))
        return self.listener.update(chunk)

    def run(self):
        self.runner.start()
        Event().wait()  # Wait forever


main = ListenScript.run_main

if __name__ == '__main__':
    main()
