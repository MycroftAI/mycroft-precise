#!/usr/bin/env python3
# Copyright (c) 2017 Mycroft AI Inc.

import sys

sys.path += ['.', 'runner']  # noqa

from threading import Event
from random import randint
from subprocess import Popen
from prettyparse import create_parser

from precise.pocketsphinx.listener import PocketsphinxListener
from precise_runner import PreciseRunner
from precise_runner.runner import ListenerEngine

usage = '''
    Run Pocketsphinx on microphone audio input
    
    :key_phrase str
        Key phrase composed of words from dictionary
    
    :dict_file str
        Filename of dictionary with word pronunciations
    
    :hmm_folder str
        Folder containing hidden markov model
    
    :-th --threshold str 1e-90
        Threshold for activations
    
    :-c --chunk-size int 2048
        Samples between inferences
'''

session_id, chunk_num = '%03d' % randint(0, 999), 0


def main():
    args = create_parser(usage).parse_args()

    def on_activation():
        Popen(['aplay', '-q', 'data/activate.wav'])

    def on_prediction(conf):
        print('!' if conf > 0.5 else '.', end='', flush=True)

    runner = PreciseRunner(
        ListenerEngine(
            PocketsphinxListener(
                args.key_phrase, args.dict_file, args.hmm_folder, args.threshold, args.chunk_size
            )
        ), 3, on_activation=on_activation, on_prediction=on_prediction
    )
    runner.start()
    Event().wait()  # Wait forever


if __name__ == '__main__':
    main()
