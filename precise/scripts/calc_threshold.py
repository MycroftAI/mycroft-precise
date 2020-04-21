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
"""
Update the threshold values of a model for a dataset.
This makes the sensitivity more accurate and linear

:model str
    Either Keras (.net) or TensorFlow (.pb) model to adjust

:input_file str
    Input stats file that was outputted from precise-graph

:-k --model-key str -
    Custom model name to use from the stats.json

:-s --smoothing float 1.2
    Amount of extra smoothing to apply

:-c --center float 0.2
    Decoded threshold that is mapped to 0.5. Proportion of
    false negatives at sensitivity=0.5
"""
from math import sqrt

from os.path import basename, splitext
from prettyparse import Usage

from precise.params import inject_params, save_params
from precise.scripts.base_script import BaseScript
from precise.stats import Stats


class CalcThresholdScript(BaseScript):
    usage = Usage(__doc__)

    def __init__(self, args):
        super().__init__(args)

    def run(self):
        args = self.args
        import numpy as np

        model_data = {
            name: Stats.from_np_dict(data) for name, data in np.load(args.input_file)['data'].item().items()
        }
        model_name = args.model_key or basename(splitext(args.model)[0])

        if model_name not in model_data:
            print("Could not find model '{}' in saved models in stats file: {}".format(model_name, list(model_data)))
            raise SystemExit(1)

        stats = model_data[model_name]

        save_spots = (stats.outputs != 0) & (stats.outputs != 1)
        if save_spots.sum() == 0:
            print('No data (or all NaN)')
            return

        stats.outputs = stats.outputs[save_spots]
        stats.targets = stats.targets[save_spots]
        inv = -np.log(1 / stats.outputs - 1)

        pos = np.extract(stats.targets > 0.5, inv)
        pos_mu = pos.mean().item()
        pos_std = sqrt(np.mean((pos - pos_mu) ** 2)) * args.smoothing

        print('Peak: {:.2f} mu, {:.2f} std'.format(pos_mu, pos_std))
        pr = inject_params(args.model)
        pr.__dict__.update(threshold_config=(
            (pos_mu, pos_std),
        ))
        save_params(args.model)
        print('Saved params to {}.params'.format(args.model))


main = CalcThresholdScript.run_main

if __name__ == '__main__':
    main()
