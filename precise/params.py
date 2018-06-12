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
from os.path import isfile

import json
from collections import namedtuple
from math import floor

import numpy as np


def _create_listener_params():
    cls = namedtuple('ListenerParams',
                     'window_t hop_t buffer_t sample_rate sample_depth n_mfcc n_filt n_fft')
    cls.buffer_samples = property(
        lambda s: s.hop_samples * (int(np.round(s.sample_rate * s.buffer_t)) // s.hop_samples)
    )
    cls.n_features = property(
        lambda s: 1 + int(floor((s.buffer_samples - s.window_samples) / s.hop_samples))
    )
    cls.window_samples = property(lambda s: int(s.sample_rate * s.window_t + 0.5))
    cls.hop_samples = property(lambda s: int(s.sample_rate * s.hop_t + 0.5))
    cls.max_samples = property(lambda s: int(s.buffer_t * s.sample_rate))
    cls.feature_size = property(lambda s: s.n_mfcc)

    return cls


class Proxy:
    def __init__(self, obj):
        self.obj = obj

    def __getattr__(self, item):
        return getattr(self.obj, item)

    def __setattr__(self, key, value):
        if key == 'obj':
            object.__setattr__(self, key, value)
        else:
            raise AttributeError('Cannot set attributes to proxy')

    def __hash__(self):
        return self.obj.__hash__()


ListenerParams = _create_listener_params()

# Reference to global listener parameters
pr = Proxy(ListenerParams(
    window_t=0.1, hop_t=0.05, buffer_t=1.5, sample_rate=16000,
    sample_depth=2, n_mfcc=13, n_filt=20, n_fft=512
))  # type: ListenerParams


def inject_params(model_name: str) -> ListenerParams:
    """Set the global listener params to a saved model"""
    params_file = model_name + '.params'
    try:
        with open(params_file) as f:
            pr.obj = ListenerParams(**json.load(f))
    except (OSError, ValueError, TypeError):
        if isfile(model_name):
            print('Warning: Failed to load parameters from ' + params_file)
    return pr


def save_params(model_name: str):
    """Save current global listener params to a file"""
    with open(model_name + '.params', 'w') as f:
        json.dump(pr._asdict(), f)
