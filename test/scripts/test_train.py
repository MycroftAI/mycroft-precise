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
from os.path import isfile

from precise.params import pr
from precise.scripts.train import TrainScript
from test.scripts.dummy_audio_folder import DummyAudioFolder


class DummyTrainFolder(DummyAudioFolder):
    def __init__(self, count=10):
        super().__init__(count)
        self.generate_samples(self.subdir('wake-word'), 'ww-{}.wav', 1.0,
                              self.rand(0, 2 * pr.buffer_t))
        self.generate_samples(self.subdir('not-wake-word'), 'nww-{}.wav', 0.0,
                              self.rand(0, 2 * pr.buffer_t))
        self.generate_samples(self.subdir('test', 'wake-word'), 'ww-{}.wav',
                              1.0, self.rand(0, 2 * pr.buffer_t))
        self.generate_samples(self.subdir('test', 'not-wake-word'),
                              'nww-{}.wav', 0.0, self.rand(0, 2 * pr.buffer_t))
        self.model = self.path('model.net')


class TestTrain:
    def test_run_basic(self):
        """Run a training and check that a model is generated."""
        folders = DummyTrainFolder(10)
        script = TrainScript.create(model=folders.model, folder=folders.root)
        script.run()
        assert isfile(folders.model)
