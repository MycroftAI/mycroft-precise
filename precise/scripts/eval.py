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
"""
Evaluate a list of models on a dataset

:-u --use-train
    Evaluate training data instead of test data

:-t --threshold float 0.5
    Network output to be considered an activation

:-pw --pocketsphinx-wake-word str -
    Optional wake word used to
    generate a Pocketsphinx data point

:-pd --pocketsphinx-dict str -
    Optional word dictionary used to
    generate a Pocketsphinx data point
    Format = wake-word.yy-mm-dd.dict

:-pf --pocketsphinx-folder str -
    Optional hmm folder used to
    generate a Pocketsphinx data point.

:-pth --pocketsphinx-threshold float 1e-90
    Optional threshold used to
    generate a Pocketsphinx data point

:-o --output str stats.json
    Output json file

...
"""
import json
from os.path import isfile, isdir
from prettyparse import Usage

from precise.network_runner import Listener
from precise.params import inject_params
from precise.pocketsphinx.scripts.test import PocketsphinxTestScript
from precise.scripts.base_script import BaseScript
from precise.stats import Stats
from precise.train_data import TrainData


class EvalScript(BaseScript):
    usage = Usage(__doc__)
    usage.add_argument('models', nargs='*',
                       help='List of model filenames in format: wake-word.yy-mm-dd.net')
    usage |= TrainData.usage

    def __init__(self, args):
        super().__init__(args)
        if not (
                bool(args.pocketsphinx_dict) ==
                bool(args.pocketsphinx_folder) ==
                bool(args.pocketsphinx_wake_word)
        ):
            raise ValueError('Must pass all or no Pocketsphinx arguments')
        self.is_pocketsphinx = bool(args.pocketsphinx_dict)

        if self.is_pocketsphinx:
            if not isfile(args.pocketsphinx_dict):
                raise ValueError('No such file: ' + args.pocketsphinx_dict)
            if not isdir(args.pocketsphinx_folder):
                raise ValueError('No such folder: ' + args.pocketsphinx_folder)

    def run(self):
        args = self.args
        data = TrainData.from_both(args.tags_file, args.tags_folder, args.folder)
        data_files = data.train_files if args.use_train else data.test_files
        print('Data:', data)

        metrics = {}

        if self.is_pocketsphinx:
            script = PocketsphinxTestScript.create(
                key_phrase=args.pocketsphinx_wake_word, dict_file=args.pocketsphinx_dict,
                hmm_folder=args.pocketsphinx_folder, threshold=args.pocketsphinx_threshold
            )
            ww_files, nww_files = data_files
            script.run_test(ww_files, 'Wake Word', 1.0)
            script.run_test(nww_files, 'Not Wake Word', 0.0)
            stats = script.get_stats()
            metrics[args.pocketsphinx_dict] = stats.to_dict(args.threshold)

        for model_name in args.models:
            print('Calculating', model_name + '...')
            inject_params(model_name)

            train, test = data.load(args.use_train, not args.use_train)
            inputs, targets = train if args.use_train else test
            predictions = Listener.find_runner(model_name)(model_name).predict(inputs)

            stats = Stats(predictions, targets, sum(data_files, []))

            print('----', model_name, '----')
            print(stats.counts_str())
            print()
            print(stats.summary_str())
            print()
            metrics[model_name] = stats.to_dict(args.threshold)

        print('Writing to:', args.output)
        with open(args.output, 'w') as f:
            json.dump(metrics, f)


main = EvalScript.run_main

if __name__ == '__main__':
    main()
