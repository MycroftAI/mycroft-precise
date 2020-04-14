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
import sys

import glob
import re
from os.path import join

from precise.scripts.engine import EngineScript
from runner.precise_runner import ReadWriteStream


class FakeStdin:
    def __init__(self, data: bytes):
        self.buffer = ReadWriteStream(data)

    def isatty(self):
        return False


class FakeStdout:
    def __init__(self):
        self.buffer = ReadWriteStream()


def test_engine(train_folder, train_script):
    """
    Test t hat the output format of the engina matches a decimal form in the
    range 0.0 - 1.0.
    """
    train_script.run()
    with open(glob.glob(join(train_folder.root, 'wake-word', '*.wav'))[0], 'rb') as f:
        data = f.read()
    try:
        sys.stdin = FakeStdin(data)
        sys.stdout = FakeStdout()
        EngineScript.create(model_name=train_folder.model).run()
        assert re.match(rb'[01]\.[0-9]+', sys.stdout.buffer.buffer)
    finally:
        sys.stdin = sys.__stdin__
        sys.stdout = sys.__stdout__
