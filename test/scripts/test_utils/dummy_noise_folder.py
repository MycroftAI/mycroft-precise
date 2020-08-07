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
import numpy as np

from test.scripts.test_utils.temp_folder import TempFolder
from test.scripts.test_utils.dummy_train_folder import DummyTrainFolder


class DummyNoiseFolder(TempFolder):
    def __init__(self):
        super().__init__()
        self.source = self.subdir('source')
        self.noise = self.subdir('noise')
        self.output = self.subdir('output')

        self.source_folder = DummyTrainFolder(root=self.source)
        self.noise_folder = DummyTrainFolder(root=self.noise)

    def generate_default(self, count=10):
        self.source_folder.generate_default(count)
        self.noise_folder.generate_samples(
            count, [], 'noise-{}.wav',
            lambda: np.ones(self.noise_folder.get_duration(), dtype=float)
        )
