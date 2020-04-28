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

from precise.scripts.train import TrainScript
from test.scripts.test_utils.dummy_train_folder import DummyTrainFolder


class TestTrain:
    def test_run_basic(self, train_folder: DummyTrainFolder):
        """Run a training and check that a model is generated."""
        train_script = TrainScript.create(model=train_folder.model, folder=train_folder.root, epochs=10)
        train_script.run()
        assert isfile(train_folder.model)
        assert isfile(train_folder.model + '.params')
