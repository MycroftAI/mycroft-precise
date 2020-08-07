import numpy as np
from precise.util import audio_to_buffer, buffer_to_audio
from precise.params import pr


def test_audio_serialization():
    audio = np.array([-1.0, 1.0] * pr.buffer_samples)
    audio2 = buffer_to_audio(audio_to_buffer(audio))
    assert np.abs(audio - audio2).max() < 1.0 / 32767.0
    audio = np.random.random(pr.buffer_samples) * 2.0 - 1.0
    audio2 = buffer_to_audio(audio_to_buffer(audio))
    assert np.abs(audio - audio2).max() < 1.0 / 32767.0
