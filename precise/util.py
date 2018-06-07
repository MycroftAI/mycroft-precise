# Copyright 2018 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import *

import numpy as np
from os.path import join, dirname, abspath

from precise.params import pr


def buffer_to_audio(buffer: bytes) -> np.ndarray:
    """Convert a raw mono audio byte string to numpy array of floats"""
    return np.fromstring(buffer, dtype='<i2').astype(np.float32, order='C') / 32768.0


def audio_to_buffer(audio: np.ndarray) -> bytes:
    """Convert a numpy array of floats to raw mono audio"""
    return (audio * 32768).astype('<i2').tostring()


def load_audio(file: Any) -> np.ndarray:
    """
    Args:
        file: Audio filename or file object
    Returns:
        samples: Sample rate and audio samples from 0..1
    """
    import wavio
    try:
        wav = wavio.read(file)
    except EOFError:
        wav = wavio.Wav(np.array([[]], dtype=np.int16), 16000, 2)
    if wav.data.dtype != np.int16:
        raise ValueError('Unsupported data type: ' + str(wav.data.dtype))
    if wav.rate != pr.sample_rate:
        raise ValueError('Unsupported sample rate: ' + str(wav.rate))

    data = np.squeeze(wav.data)
    return data.astype(np.float32) / float(np.iinfo(data.dtype).max)


def save_audio(filename: str, audio: np.ndarray):
    import wavio
    save_audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
    wavio.write(filename, save_audio, pr.sample_rate, sampwidth=pr.sample_depth, scale='none')

def play_audio(filename: str):
    """
    Args:
        filename: Audio filename
    """
    import platform
    from subprocess import Popen

    player = 'play' if platform.system() == 'Darwin' else 'aplay'
    Popen([player, '-q', filename])

def activate_notify():
    audio = 'data/activate.wav'
    audio = abspath(dirname(abspath(__file__)) + '/../' + audio)

    play_audio(audio)

def glob_all(folder: str, filt: str) -> List[str]:
    """Recursive glob"""
    import os
    import fnmatch
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, filt):
            matches.append(os.path.join(root, filename))
    return matches


def find_wavs(folder: str) -> Tuple[List[str], List[str]]:
    """Finds wake-word and not-wake-word wavs in folder"""
    return (glob_all(join(folder, 'wake-word'), '*.wav'),
            glob_all(join(folder, 'not-wake-word'), '*.wav'))
