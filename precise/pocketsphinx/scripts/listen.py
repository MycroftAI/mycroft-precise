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
Run Pocketsphinx on microphone audio input

:key_phrase str
    Key phrase composed of words from dictionary

:dict_file str
    Filename of dictionary with word pronunciations

:hmm_folder str
    Folder containing hidden markov model

:-th --threshold str 1e-90
    Threshold for activations

:-c --chunk-size int 2048
    Samples between inferences
"""
from precise_runner import PreciseRunner
from precise_runner.runner import ListenerEngine
from prettyparse import Usage
from threading import Event

from precise.pocketsphinx.listener import PocketsphinxListener
from precise.scripts.base_script import BaseScript
from precise.util import activate_notify


class PocketsphinxListenScript(BaseScript):
    usage = Usage(__doc__)

    def run(self):
        def on_activation():
            activate_notify()

        def on_prediction(conf):
            print('!' if conf > 0.5 else '.', end='', flush=True)

        args = self.args
        runner = PreciseRunner(
            ListenerEngine(
                PocketsphinxListener(
                    args.key_phrase, args.dict_file, args.hmm_folder, args.threshold, args.chunk_size
                )
            ), 3, on_activation=on_activation, on_prediction=on_prediction
        )
        runner.start()
        Event().wait()  # Wait forever


main = PocketsphinxListenScript.run_main

if __name__ == '__main__':
    main()
