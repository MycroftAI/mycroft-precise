#!/usr/bin/env python3
# Copyright (c) 2017 Mycroft AI Inc.

import sys

sys.path += ['.', 'runner']  # noqa

from argparse import ArgumentParser
from psutil import Popen
from precise_runner import PreciseRunner
from threading import Event


def main():
    parser = ArgumentParser('Implementation demo of precise-stream')
    parser.add_argument('-m', '--model', default='keyword.pb')
    args = parser.parse_args()

    def on_prediction(prob):
        print('!' if prob > 0.5 else '.', end='', flush=True)

    def on_activation():
        Popen(['aplay', 'data/activate.wav'])

    PreciseRunner('./precise/stream.py', args.model, on_prediction=on_prediction, on_activation=on_activation).start()
    Event().wait()  # Wait forever

if __name__ == '__main__':
    main()

