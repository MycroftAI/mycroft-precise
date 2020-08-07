#!/usr/bin/env python3
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
import atexit

import os
from os import makedirs
from os.path import isdir, join
from shutil import rmtree
from tempfile import mkdtemp


class TempFolder:
    def __init__(self, root=None):
        self.root = mkdtemp() if root is None else root
        atexit.register(self.cleanup)

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
