# Python 3

from collections import namedtuple
from math import floor


def _make_cls():
    cls = namedtuple('ListenerParams', 'window_t hop_t buffer_t sample_rate sample_depth n_mfcc n_filt n_fft')

    def add_prop(name, fn):
        setattr(cls, name, property(fn))
    import numpy as np

    add_prop('buffer_samples', lambda s: s.hop_samples * (int(np.round(s.sample_rate * s.buffer_t)) // s.hop_samples))
    add_prop('window_samples', lambda s: int(np.round(s.sample_rate * s.window_t)))
    add_prop('hop_samples', lambda s: int(np.round(s.sample_rate * s.hop_t)))

    add_prop('n_features', lambda s: 1 + int(floor((s.buffer_samples - s.window_samples) / s.hop_samples)))
    add_prop('feature_size', lambda s: s.n_mfcc)
    add_prop('max_samples', lambda s: int(s.buffer_t * s.sample_rate))
    return cls


ListenerParams = _make_cls()
