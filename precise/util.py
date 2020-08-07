# Copyright 2019 Mycroft AI Inc.
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
"""
Miscellaneous utility functions for things like audio loading
"""
import hashlib
import numpy as np
from os.path import join, dirname, abspath
from typing import *

from precise.params import pr


class InvalidAudio(ValueError):
    """Thrown the audio isn't in the expected format"""
    pass


def chunk_audio(audio: np.ndarray, chunk_size: int) -> Generator[np.ndarray, None, None]:
    for i in range(chunk_size, len(audio), chunk_size):
        yield audio[i - chunk_size:i]


def float_audio_to_int(audio: np.ndarray) -> np.ndarray:
    """Converts [-1.0, 1.0] -> [-32768,  32767]"""
    return (audio.astype(np.float32, order='C') * (0x7FFF + 0.5) - 0.5).astype('<i2')


def int_audio_to_float(int_audio: np.ndarray) -> np.ndarray:
    """Converts [-32768,  32767] -> [-1.0, 1.0]"""
    return (int_audio + 0.5) / (0x7FFF + 0.5)


def buffer_to_audio(buffer: bytes) -> np.ndarray:
    """Convert a raw mono audio byte string to numpy array of floats"""
    return int_audio_to_float(np.frombuffer(buffer, dtype='<i2'))


def audio_to_buffer(audio: np.ndarray) -> bytes:
    """Convert a numpy array of floats to raw mono audio"""
    return float_audio_to_int(audio).tostring()


def load_audio(file: Any) -> np.ndarray:
    """
    Loads properly formatted audio from a file to a numpy array
    Args:
        file: Audio filename or file object
    Returns:
        samples: Sample rate and audio samples from 0..1
    """
    import wavio
    import wave
    try:
        wav = wavio.read(file)
    except (EOFError, wave.Error):
        wav = wavio.Wav(np.array([[]], dtype=np.int16), 16000, 2)
    if wav.data.dtype != np.int16:
        raise InvalidAudio('Unsupported data type: ' + str(wav.data.dtype))
    if wav.rate != pr.sample_rate:
        raise InvalidAudio('Unsupported sample rate: ' + str(wav.rate))

    return int_audio_to_float(np.squeeze(wav.data))


def save_audio(filename: str, audio: np.ndarray):
    """Save loaded audio to file using the configured audio parameters"""
    import wavio
    int_audio = float_audio_to_int(audio)
    wavio.write(filename, int_audio, pr.sample_rate, sampwidth=pr.sample_depth, scale='none')


def play_audio(filename: str):
    """
    Args:
        filename: Audio filename
    """
    import platform
    from subprocess import Popen

    if platform.system() == 'Darwin':
        Popen(['afplay', filename])
    else:
        Popen(['aplay', '-q', filename])


def activate_notify():
    """Play some sound to indicate a wakeword activation when testing a model"""
    audio = 'data/activate.wav'
    audio = join(dirname(abspath(__file__)), audio)
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


def calc_sample_hash(inp: np.ndarray, outp: np.ndarray) -> str:
    """Hashes a training sample of an input vector and target output vector"""
    md5 = hashlib.md5()
    md5.update(inp.tostring())
    md5.update(outp.tostring())
    return md5.hexdigest()
