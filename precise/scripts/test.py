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
Test a model against a dataset

:model str
    Either Keras (.net) or TensorFlow (.pb) model to test

:-u --use-train
    Evaluate training data instead of test data

:-nf --no-filenames
    Don't print out the names of files that failed

:-t --threshold float 0.5
    Network output required to be considered an activation

...
"""
from prettyparse import Usage

from precise.network_runner import Listener
from precise.params import inject_params
from precise.scripts.base_script import BaseScript
from precise.stats import Stats
from precise.train_data import TrainData


class TestScript(BaseScript):
    usage = Usage(__doc__) | TrainData.usage

    def run(self):
        args = self.args
        inject_params(args.model)
        data = TrainData.from_both(args.tags_file, args.tags_folder, args.folder)
        train, test = data.load(args.use_train, not args.use_train, shuffle=False)
        inputs, targets = train if args.use_train else test

        filenames = sum(data.train_files if args.use_train else data.test_files, [])
        predictions = Listener.find_runner(args.model)(args.model).predict(inputs)
        stats = Stats(predictions, targets, filenames)

        print('Data:', data)

        if not args.no_filenames:
            fp_files = stats.calc_filenames(False, True, args.threshold)
            fn_files = stats.calc_filenames(False, False, args.threshold)
            print('=== False Positives ===')
            print('\n'.join(fp_files))
            print()
            print('=== False Negatives ===')
            print('\n'.join(fn_files))
            print()
        print(stats.counts_str(args.threshold))
        print()
        print(stats.summary_str(args.threshold))


main = TestScript.run_main

if __name__ == '__main__':
    main()
