from typing import *

import numpy as np

from precise.params import pr


def buffer_to_audio(buffer: bytes) -> np.ndarray:
    """Convert a raw mono audio byte string to numpy array of floats"""
    return np.fromstring(buffer, dtype='<i2').astype(np.float32, order='C') / 32768.0


def load_audio(file: Any) -> np.ndarray:
    """
    Args:
        file: Audio filename or file object
    Returns:
        samples: Sample rate and audio samples from 0..1
    """
    import wavio
    wav = wavio.read(file)
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
    """Finds keyword and not-keyword wavs in folder"""
    return glob_all(folder + '/keyword', '*.wav'), glob_all(folder + '/not-keyword', '*.wav')
