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
# limitations under the License

from os.path import isfile

from precise.scripts.calc_threshold import CalcThresholdScript
from precise.scripts.eval import EvalScript
from precise.scripts.graph import GraphScript
from test.scripts.test_utils.dummy_train_folder import DummyTrainFolder


def read_content(filename):
    with open(filename) as f:
        return f.read()


def test_combined(train_folder: DummyTrainFolder, trained_model: str):
    """Test a "normal" development cycle, train, evaluate and calc threshold.
    """
    params_file = trained_model + '.params'
    assert isfile(trained_model)
    assert isfile(params_file)

    EvalScript.create(folder=train_folder.root,
                      models=[trained_model]).run()

    # Ensure that the graph script generates a numpy savez file
    out_file = train_folder.path('outputs.npz')
    graph_script = GraphScript.create(folder=train_folder.root,
                                      models=[trained_model],
                                      output_file=out_file)
    graph_script.run()
    assert isfile(out_file)

    # Esure the params are updated after threshold is calculated
    params_before = read_content(params_file)
    CalcThresholdScript.create(folder=train_folder.root,
                               model=trained_model,
                               input_file=out_file).run()
    assert params_before != read_content(params_file)
