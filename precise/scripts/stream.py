#!/usr/bin/env python3
# Copyright (c) 2017 Mycroft AI Inc.

import sys

sys.path += ['.']  # noqa

import os
from prettyparse import create_parser

from precise.network_runner import Listener
from precise import __version__

usage = '''
    stdin should be a stream of raw int16 audio, written in
    groups of CHUNK_SIZE samples. If no CHUNK_SIZE is given
    it will read until EOF. For every chunk, an inference
    will be given via stdout as a float string, one per line
    
    :model_name str
        Keras or Tensorflow model to read from

    ...
'''


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    stdout = sys.stdout
    sys.stdout = sys.stderr

    parser = create_parser(usage)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('chunk_size', type=int, nargs='?', default=-1,
                        help='Number of samples to read before making a prediction.'
                             'Higher values are less computationally expensive')
    parser.usage = parser.format_usage().strip().replace('usage: ', '') + ' < audio.wav'
    args = parser.parse_args()

    if sys.stdin.isatty():
        parser.error('Please pipe audio via stdin using < audio.wav')

    listener = Listener(args.model_name, args.chunk_size)

    try:
        while True:
            conf = listener.update(sys.stdin.buffer)
            stdout.buffer.write((str(conf) + '\n').encode('ascii'))
            stdout.buffer.flush()
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == '__main__':
    main()
