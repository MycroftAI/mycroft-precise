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
from glob import glob
from os.path import join

import numpy as np
from precise_runner import ReadWriteStream

from precise.params import pr
from precise.scripts.listen import ListenScript
from precise.util import audio_to_buffer
from test.scripts.test_utils.temp_folder import TempFolder


class TestListen:
    def test_listen(self, trained_model: str, temp_folder: TempFolder):
        """Run the trained model on input"""
        activations_folder = temp_folder.subdir('activations')
        script = ListenScript.create(model=trained_model, save_dir=activations_folder)
        script.runner.stream = stream = ReadWriteStream()

        script.runner.start()
        stream.write(audio_to_buffer(np.random.random(5 * pr.sample_rate) * 2 - 1))  # Write silence
        stream.write(audio_to_buffer(np.ones(2 * pr.sample_rate, dtype=float) * 2 - 1))  # Write wake word
        stream.write(audio_to_buffer(np.random.random(5 * pr.sample_rate) * 2 - 1))  # Write more silence
        script.runner.stop()

        assert len(glob(join(activations_folder, '*.wav'))) == 1
