#!/usr/bin/env python3
# Copyright (c) 2017 Mycroft AI Inc.

import sys

sys.path += ['.']  # noqa

from prettyparse import create_parser
from precise.params import inject_params
from precise.model import load_precise_model
from precise.train_data import TrainData

usage = '''
    Test a model against a dataset
    
    :model str
        Keras model file (.net) to test
    
    :-t --use-train
        Evaluate training data instead of test data
    
    :-nf --no-filenames
        Don't print out the names of files that failed
    
    ...
'''


def show_stats(false_pos, false_neg, true_pos, true_neg, show_filenames):
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


def main():
    args = TrainData.parse_args(create_parser(usage))

    inject_params(args.model)

    data = TrainData.from_both(args.db_file, args.db_folder, args.data_dir)
    train, test = data.load(args.use_train, not args.use_train)
    inputs, targets = train if args.use_train else test

    filenames = sum(data.train_files if args.use_train else data.test_files, [])
    predictions = load_precise_model(args.model).predict(inputs)

    true_pos, true_neg = [], []
    false_pos, false_neg = [], []

    for name, target, prediction in zip(filenames, targets, predictions):
        {
            (True, False): false_pos,
            (True, True): true_pos,
            (False, True): false_neg,
            (False, False): true_neg
        }[prediction[0] > 0.5, target[0] > 0.5].append(name)

    print('Data:', data)
    show_stats(false_pos, false_neg, true_pos, true_neg, not args.no_filenames)


if __name__ == '__main__':
    main()
