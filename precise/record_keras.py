#!/usr/bin/env python3
# Copyright (c) 2017 Mycroft AI Inc.

import sys
sys.path += ['.']

from io import StringIO

silent = False
should_save = False
enable_sound = False
save_prefix = ''
num_to_save = -1
save_act = False

act_lev = 0
was_active = False


while len(sys.argv) > 1:
    a = sys.argv[1]
    del sys.argv[1]
    if a == 'save':
        should_save = True
        while len(sys.argv) > 1:
            a = sys.argv[1]
            if a.isdigit():
                num_to_save = int(a)
    elif a == 'sound':
        enable_sound = True
    elif a == 'silent':
        silent = True
    else:
        break

if silent:
    _stdout = sys.stdout
    sys.stdout = StringIO()  # capture any output

import wave
from random import randint

session_id = randint(0, 1000)
chunk_id = 0

from precise.common import *
from pyaudio import PyAudio, get_format_from_width
from os.path import join
import numpy as np


class PreciseRecognizer:
    def __init__(self):
        self.model = load_precise_model('lstm-tanh-less-fp-2.net')

    @staticmethod
    def buffer_to_audio(buffer: bytearray) -> np.array:
        """Convert a raw mono audio byte string to numpy array of floats"""
        return np.fromstring(str(buffer), dtype='<i2').astype(np.float32, order='C') / 32768.0

    def found_wake_word(self, raw_data):
        inp = vectorize(self.buffer_to_audio(raw_data))
        return self.model.predict_on_batch(inp[np.newaxis]) >= 0.5

CHANNELS = 1
CHUNK_SIZE = 1024
RATE = pr.sample_rate
WIDTH = 2  # Int16
FORMAT = get_format_from_width(WIDTH)
BUFFER_LEN = WIDTH * CHANNELS * pr.max_samples

p = PyAudio()
recognizer = PreciseRecognizer()
stream = p.open(RATE, CHANNELS, FORMAT, True, frames_per_buffer=CHUNK_SIZE)
buffer = b'\0' * BUFFER_LEN


def save(buffer: bytearray, debug=False):
    if not should_save:
        return

    global chunk_id, num_to_save
    nm = join('data', 'not-keyword',
              save_prefix + str(session_id) + '.' + str(chunk_id) + '.wav')
    chunk_id += 1
    num_to_save -= 1

    with wave.open(nm, 'w') as wf:
        wf.setnchannels(CHANNELS)
        wf.setframerate(RATE)
        wf.setsampwidth(WIDTH)
        wf.writeframes(buffer)
    if debug:
        print('Saved to ' + nm + '.')

start_delay = 40
print('Filling buffer...')
try:
    while True:
        chunk = stream.read(CHUNK_SIZE)
        buffer = buffer[max(0, len(buffer) - BUFFER_LEN + len(chunk)):] + chunk

        if start_delay > 0:
            start_delay -= 1
            continue
        found = recognizer.found_wake_word(buffer)

        if found:
            if not was_active:
                was_active = True
                if silent:
                    sys.stdout = _stdout
                    print(':activate:')
                    sys.stdout = StringIO()
                else:
                    print('Activate!')
                save(buffer, debug=True)    
        else:
            if was_active:
                was_active = False
            print('.', end='', flush=True)
        if num_to_save == 0:
            break

finally:
    stream.stop_stream()
    p.terminate()
