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
from precise.scripts.add_noise import AddNoiseScript

from test.scripts.dummy_audio_folder import DummyAudioFolder


class DummyNoiseFolder(DummyAudioFolder):
    def __init__(self, count=10):
        super().__init__(count)
        self.source = self.subdir('source')
        self.noise = self.subdir('noise')
        self.output = self.subdir('output')

        self.generate_samples(self.subdir('source', 'wake-word'), 'ww-{}.wav', 1.0, self.rand(0, 2))
        self.generate_samples(self.subdir('source', 'not-wake-word'), 'nww-{}.wav', 0.0, self.rand(0, 2))
        self.generate_samples(self.noise, 'noise-{}.wav', 0.5, self.rand(10, 20))


class TestAddNoise:
    def get_base_data(self, count):
        folders = DummyNoiseFolder(count)
        base_args = dict(
            folder=folders.source, noise_folder=folders.noise,
            output_folder=folders.output
        )
        return folders, base_args

    def test_run_basic(self):
        folders, base_args = self.get_base_data(10)
        script = AddNoiseScript.create(inflation_factor=1, **base_args)
        script.run()
        assert folders.count_files(folders.output) == 20

    def test_run_basic_2(self):
        folders, base_args = self.get_base_data(10)
        script = AddNoiseScript.create(inflation_factor=2, **base_args)
        script.run()
        assert folders.count_files(folders.output) == 40
