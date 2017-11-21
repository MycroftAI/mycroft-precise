#!/usr/bin/env python3

import sys

sys.path += ['.']  # noqa

from argparse import ArgumentParser

from precise.common import *


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', default='keyword.net')
    parser.add_argument('-t', '--test-dir', default='data/test')
    parser.set_defaults(load=True, save_best=True)
    args = parser.parse_args()

    filenames = sum(find_wavs(args.test_dir), [])
    inputs, outputs = load_data(args.test_dir)
    predictions = load_precise_model(args.model).predict(inputs)

    num_correct = 0
    false_pos, false_neg = [], []
    for name, correct, prediction in zip(filenames, outputs, predictions):
        if prediction < 0.5 < correct:
            false_neg += [name]
        elif prediction > 0.5 > correct:
            false_pos += [name]
        else:
            num_correct += 1

    def prc(a, b):  # Rounded percent
        return round(100.0 * a / b, 2)

    print('=== False Positives ===')
    for i in false_pos:
        print(i)
    print()
    print('=== False Negatives ===')
    for i in false_neg:
        print(i)
    print()
    print('=== Summary ===')
    total = num_correct + len(false_pos) + len(false_neg)
    print(num_correct, "out of", total)
    print(prc(num_correct, total), "%")
    print()
    print(prc(len(false_pos), total), "% false positives")
    print(prc(len(false_neg), total), "% false negatives")


if __name__ == '__main__':
    main()
