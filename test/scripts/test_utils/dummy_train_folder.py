# Copyright 2020 Mycroft AI Inc.
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
import random
from os.path import join

import numpy as np

from precise.params import pr
from precise.util import save_audio
from test.scripts.test_utils.temp_folder import TempFolder


class DummyTrainFolder(TempFolder):
    def __init__(self, root=None):
        super().__init__(root)
        self.model = self.path('model.net')

    def generate_samples(self, count, subfolder, name, generator):
        """
        Generate sample audio files in a folder

        The file is generated in the specified folder, with the specified
        name and generated value.

        Args:
            count: Number of samples to generate
            subfolder: String or list of subfolder path
            name: Format string used to generate each sample
            generator: Function called to get the data for each sample
        """
        if isinstance(subfolder, str):
            subfolder = [subfolder]
        for i in range(count):
            save_audio(join(self.subdir(*subfolder), name.format(i)), generator())

    def get_duration(self):
        """Generate a random sample duration"""
        return int(random.random() * 2 * pr.buffer_samples)

    def generate_default(self, count=10):
        self.generate_samples(
            count, 'wake-word', 'ww-{}.wav',
            lambda: np.ones(self.get_duration(), dtype=float)
        )
        self.generate_samples(
            count, 'not-wake-word', 'nww-{}.wav',
            lambda: np.random.random(self.get_duration()) * 2 - 1
        )
        self.generate_samples(
            count, ('test', 'wake-word'), 'ww-{}.wav',
            lambda: np.ones(self.get_duration(), dtype=float)
        )
        self.generate_samples(
            count, ('test', 'not-wake-word'), 'nww-{}.wav',
            lambda: np.random.random(self.get_duration()) * 2 - 1
        )
        self.model = self.path('model.net')