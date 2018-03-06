#!/usr/bin/env python3
# Copyright 2018 Mycroft AI Inc.
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
import json

from prettyparse import create_parser

from precise.network_runner import Listener
from precise.params import inject_params
from precise.scripts.test import show_stats
from precise.train_data import TrainData

usage = '''
    Evaluate a list of models on a dataset
    
    :-t --use-train
        Evaluate training data instead of test data
    
    :-o --output str stats.json
        Output json file
    
    ...
'''


def main():
    parser = create_parser(usage)
    parser.add_argument('models', nargs='*', help='List of model filenames')
    args = TrainData.parse_args(parser)

    data = TrainData.from_both(args.db_file, args.db_folder, args.data_dir)
    filenames = sum(data.train_files if args.use_train else data.test_files, [])
    print('Data:', data)

    stats = {}

    for model_name in args.models:
        print('Calculating', model_name + '...')
        inject_params(model_name)

        train, test = data.load(args.use_train, not args.use_train)
        inputs, targets = train if args.use_train else test
        predictions = Listener.find_runner(model_name)(model_name).predict(inputs)

        true_pos, true_neg = [], []
        false_pos, false_neg = [], []

        for name, target, prediction in zip(filenames, targets, predictions):
            {
                (True, False): false_pos,
                (True, True): true_pos,
                (False, True): false_neg,
                (False, False): true_neg
            }[prediction[0] > 0.5, target[0] > 0.5].append(name)

        print('----', model_name, '----')
        show_stats(false_pos, false_neg, true_pos, true_neg, False)
        stats[model_name] = {
            'true_pos': len(true_pos),
            'true_neg': len(true_neg),
            'false_pos': len(false_pos),
            'false_neg': len(false_neg),
        }

    print('Writing to:', args.output)
    with open(args.output, 'w') as f:
        json.dump(stats, f)


if __name__ == '__main__':
    main()
