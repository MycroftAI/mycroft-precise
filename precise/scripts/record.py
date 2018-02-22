#!/usr/bin/env python3
# Copyright (c) 2017 Mycroft AI Inc.
from os.path import join
from random import randint
from subprocess import Popen
from threading import Event

import numpy as np
from prettyparse import create_parser

from precise.network_runner import Listener
from precise.util import save_audio, buffer_to_audio
from precise_runner import PreciseRunner
from precise_runner.runner import ListenerEngine

usage = '''
    Run a model on microphone audio input
    
    :model str
        Either Keras (.net) or Tensorflow (.pb) model to run
    
    :-c --chunk-size int 2048
        Samples between inferences
    
    :-s --save-dir str -
        Folder to save false positives
    
    :-p --save-prefix str -
        Prefix for saved filenames
'''

session_id, chunk_num = '%03d' % randint(0, 999), 0


def main():
    args = create_parser(usage).parse_args()

    def on_activation():
        Popen(['aplay', '-q', 'data/activate.wav'])
        if args.save_dir:
            global chunk_num
            nm = join(args.save_dir, args.save_prefix + session_id + '.' + str(chunk_num) + '.wav')
            save_audio(nm, audio_buffer)
            print()
            print('Saved to ' + nm + '.')
            chunk_num += 1

    def on_prediction(conf):
        print('!' if conf > 0.5 else '.', end='', flush=True)

    listener = Listener(args.model, args.chunk_size)
    audio_buffer = np.zeros(listener.pr.buffer_samples, dtype=float)

    def get_prediction(chunk):
        nonlocal audio_buffer
        audio = buffer_to_audio(chunk)
        audio_buffer = np.concatenate((audio_buffer[len(audio):], audio))
        return listener.update(chunk)

    engine = ListenerEngine(listener)
    engine.get_prediction = get_prediction
    runner = PreciseRunner(engine, 3, on_activation=on_activation, on_prediction=on_prediction)
    runner.start()
    Event().wait()  # Wait forever


if __name__ == '__main__':
    main()
