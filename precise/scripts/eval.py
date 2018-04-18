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

from os.path import isfile, isdir
from prettyparse import create_parser

from precise.network_runner import Listener
from precise.params import inject_params
from precise.pocketsphinx.listener import PocketsphinxListener
from precise.pocketsphinx.scripts.test import test_pocketsphinx
from precise.scripts.test import show_stats, calc_stats, stats_to_dict
from precise.train_data import TrainData

usage = '''
    Evaluate a list of models on a dataset
    
    :-t --use-train
        Evaluate training data instead of test data
    
    :-pw --pocketsphinx-wake-word str -
        Optional wake word used to
        generate a Pocketsphinx data point
    
    :-pd --pocketsphinx-dict str -
        Optional word dictionary used to
        generate a Pocketsphinx data point
        Format: wake-word.yy-mm-dd.dict
    
    :-pf --pocketsphinx-folder str -
        Optional hmm folder used to
        generate a Pocketsphinx data point.
    
    :-pth --pocketsphinx-threshold float 1e-90
        Optional threshold used to
        generate a Pocketsphinx data point
    
    :-o --output str stats.json
        Output json file
    
    ...
'''


def main():
    parser = create_parser(usage)
    parser.add_argument('models', nargs='*',
                        help='List of model filenames in format: wake-word.yy-mm-dd.net')
    args = TrainData.parse_args(parser)
    if not (
            bool(args.pocketsphinx_dict) ==
            bool(args.pocketsphinx_folder) ==
            bool(args.pocketsphinx_wake_word)
    ):
        parser.error('Must pass all or no Pocketsphinx arguments')

    data = TrainData.from_both(args.tags_file, args.tags_folder, args.folder)
    data_files = data.train_files if args.use_train else data.test_files
    print('Data:', data)

    metrics = {}

    if args.pocketsphinx_dict and args.pocketsphinx_folder and args.pocketsphinx_wake_word:
        if not isfile(args.pocketsphinx_dict):
            parser.error('No such file: ' + args.pocketsphinx_dict)
        if not isdir(args.pocketsphinx_folder):
            parser.error('No such folder: ' + args.pocketsphinx_folder)
        listener = PocketsphinxListener(
            args.pocketsphinx_wake_word, args.pocketsphinx_dict,
            args.pocketsphinx_folder, args.pocketsphinx_threshold
        )
        stats = test_pocketsphinx(listener, data_files)
        metrics[args.pocketsphinx_dict] = stats_to_dict(stats)

    for model_name in args.models:
        print('Calculating', model_name + '...')
        inject_params(model_name)

        train, test = data.load(args.use_train, not args.use_train)
        inputs, targets = train if args.use_train else test
        predictions = Listener.find_runner(model_name)(model_name).predict(inputs)
        stats = calc_stats(sum(data_files, []), targets, predictions)

        print('----', model_name, '----')
        show_stats(stats, False)
        metrics[model_name] = stats_to_dict(stats)

    print('Writing to:', args.output)
    with open(args.output, 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    main()
