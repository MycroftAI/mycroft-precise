#!/usr/bin/env python3
# Copyright (c) 2017 Mycroft AI Inc.

import sys
sys.path += ['.']  # noqa

from argparse import ArgumentParser
import json

from precise.common import *


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', default='keyword.net')
    parser.add_argument('-d', '--data-dir', default='data')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-l', '--load', dest='load', action='store_true')
    parser.add_argument('-nl', '--no-load', dest='load', action='store_false')
    parser.add_argument('-b', '--save-best', dest='save_best', action='store_true')
    parser.add_argument('-nb', '--no-save-best', dest='save_best', action='store_false')
    parser.set_defaults(load=True, save_best=True)
    args = parser.parse_args()

    inputs, outputs = load_data(args.data_dir)
    val_in, val_out = load_data(args.data_dir + '/test')

    print('Inputs shape:', inputs.shape)
    print('Outputs shape:', outputs.shape)
    print('Test inputs shape:', val_in.shape)
    print('Test outputs shape:', val_out.shape)

    if (0 in inputs.shape or 0 in outputs.shape or
        0 in val_in.shape or 0 in val_out.shape):
        print('Not enough data to train')
        exit(1)

    model = create_model(args.model, args.load)

    with open(args.model + '.params', 'w') as f:
        json.dump(pr._asdict(), f)

    from keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(args.model, monitor='val_acc', save_best_only=args.save_best, mode='max')

    try:
        model.fit(inputs, outputs, 5000, args.epochs, validation_data=(val_in, val_out), callbacks=[checkpoint])
    except KeyboardInterrupt:
        print()

if __name__ == '__main__':
    main()
