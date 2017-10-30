#!/usr/bin/env python3

import sys
sys.path += ['.']  # noqa

import argparse
import json

from precise.common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-l', '--load', dest='load', action='store_true')
    parser.add_argument('-nl', '--no-load', dest='load', action='store_false')
    parser.add_argument('-b', '--save-best', dest='save_best', action='store_true')
    parser.add_argument('-nb', '--no-save-best', dest='save_best', action='store_false')
    parser.set_defaults(load=True, save_best=True)
    args = parser.parse_args()

    inputs, outputs = load_data('data')
    validation_data = load_data('data/test')

    print('Inputs shape:', inputs.shape)
    print('Outputs shape:', outputs.shape)

    model = create_model('keyword.net', args.load)

    with open('keyword.net.params', 'w') as f:
        json.dump(pr._asdict(), f)

    from keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint('keyword.net', monitor='val_acc', save_best_only=args.save_best, mode='max')

    try:
        model.fit(inputs, outputs, 5000, args.epochs, validation_data=validation_data, callbacks=[checkpoint])
    except KeyboardInterrupt:
        print()

if __name__ == '__main__':
    main()
