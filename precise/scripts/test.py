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
from collections import namedtuple

from prettyparse import create_parser

from precise.model import load_precise_model
from precise.network_runner import Listener
from precise.params import inject_params
from precise.train_data import TrainData

usage = '''
    Test a model against a dataset
    
    :model str
        Either Keras (.net) or TensorFlow (.pb) model to test
    
    :-t --use-train
        Evaluate training data instead of test data
    
    :-nf --no-filenames
        Don't print out the names of files that failed
    
    ...
'''

Stats = namedtuple('Stats', 'false_pos false_neg true_pos true_neg')


def stats_to_dict(stats: Stats) -> dict:
    return {
        'true_pos': len(stats.true_pos),
        'true_neg': len(stats.true_neg),
        'false_pos': len(stats.false_pos),
        'false_neg': len(stats.false_neg),
    }


def show_stats(stats: Stats, show_filenames):
    false_pos, false_neg, true_pos, true_neg = stats

    num_correct = len(true_pos) + len(true_neg)
    total = num_correct + len(false_pos) + len(false_neg)

    def prc(a: int, b: int):  # Rounded percent
        return round(100.0 * (b and a / b), 2)

    if show_filenames:
        print('=== False Positives ===')
        for i in false_pos:
            print(i)
        print()
        print('=== False Negatives ===')
        for i in false_neg:
            print(i)
        print()
    print('=== Counts ===')
    print('False Positives:', len(false_pos))
    print('True Negatives:', len(true_neg))
    print('False Negatives:', len(false_neg))
    print('True Positives:', len(true_pos))
    print()
    print('=== Summary ===')
    print(num_correct, "out of", total)
    print(prc(num_correct, total), "%")
    print()
    print(prc(len(false_pos), len(false_pos) + len(true_neg)), "% false positives")
    print(prc(len(false_neg), len(false_neg) + len(true_pos)), "% false negatives")


def calc_stats(filenames, targets, predictions) -> Stats:
    stats = Stats([], [], [], [])
    for name, target, prediction in zip(filenames, targets, predictions):
        {
            (True, False): stats.false_pos,
            (True, True): stats.true_pos,
            (False, True): stats.false_neg,
            (False, False): stats.true_neg
        }[prediction[0] > 0.5, target[0] > 0.5].append(name)
    return stats


def main():
    args = TrainData.parse_args(create_parser(usage))

    inject_params(args.model)

    data = TrainData.from_both(args.tags_file, args.tags_folder, args.folder)
    train, test = data.load(args.use_train, not args.use_train)
    inputs, targets = train if args.use_train else test

    filenames = sum(data.train_files if args.use_train else data.test_files, [])
    predictions = Listener.find_runner(args.model)(args.model).predict(inputs)
    stats = calc_stats(filenames, targets, predictions)

    print('Data:', data)
    show_stats(stats, not args.no_filenames)


if __name__ == '__main__':
    main()
