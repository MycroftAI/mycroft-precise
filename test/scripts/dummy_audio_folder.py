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
import atexit

import numpy as np
import os
from os import makedirs
from os.path import isdir, join
from shutil import rmtree
from tempfile import mkdtemp

from precise.params import pr
from precise.util import save_audio


class DummyAudioFolder:
    def __init__(self, count=10):
        self.count = count
        self.root = mkdtemp()
        atexit.register(self.cleanup)

    def rand(self, min, max):
        return min + (max - min) * np.random.random() * pr.buffer_t

    def generate_samples(self, folder, name, value, duration):
        """Generate sample file.

        The file is generated in the specified folder, with the specified name,
        dummy value and duration.
        """
        for i in range(self.count):
            save_audio(join(folder, name.format(i)),
                       np.array([value] * int(duration * pr.sample_rate)))

    def subdir(self, *parts):
        folder = self.path(*parts)
        if not isdir(folder):
            makedirs(folder)
        return folder

    def path(self, *path):
        return join(self.root, *path)

    def count_files(self, folder):
        return sum([len(files) for r, d, files in os.walk(folder)])

    def cleanup(self):
        if isdir(self.root):
            rmtree(self.root)
