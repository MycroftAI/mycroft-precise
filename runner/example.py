#!/usr/bin/env python3
# Copyright (c) 2017 Mycroft AI Inc.
from argparse import ArgumentParser
from subprocess import Popen
from precise_runner import PreciseRunner, PreciseEngine
from threading import Event


def main():
    parser = ArgumentParser('Implementation demo of precise-engine')
    parser.add_argument('model')
    args = parser.parse_args()

    def on_prediction(prob):
        print('!' if prob > 0.5 else '.', end='', flush=True)

    def on_activation():
        Popen(['aplay', '-q', 'data/activate.wav'])

    engine = PreciseEngine('./precise/engine.py', args.model)
    PreciseRunner(engine, on_prediction=on_prediction, on_activation=on_activation,
                  trigger_level=0).start()
    Event().wait()  # Wait forever


if __name__ == '__main__':
    main()
