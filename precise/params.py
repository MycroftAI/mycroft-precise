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
from math import floor

import attr
import json
import hashlib
from os.path import isfile


@attr.s(frozen=True)
class ListenerParams:
    window_t = attr.ib()  # type: float
    hop_t = attr.ib()  # type: float
    buffer_t = attr.ib()  # type: float
    sample_rate = attr.ib()  # type: int
    sample_depth = attr.ib()  # type: int
    n_mfcc = attr.ib()  # type: int
    n_filt = attr.ib()  # type: int
    n_fft = attr.ib()  # type: int
    use_delta = attr.ib()  # type: bool
    vectorizer = attr.ib()  # type: int
    threshold_config = attr.ib()  # type: tuple
    threshold_center = attr.ib()  # type: float

    @property
    def buffer_samples(self):
        samples = int(self.sample_rate * self.buffer_t + 0.5)
        return self.hop_samples * (samples // self.hop_samples)

    @property
    def n_features(self):
        return 1 + int(floor((self.buffer_samples - self.window_samples) / self.hop_samples))

    @property
    def window_samples(self):
        return int(self.sample_rate * self.window_t + 0.5)

    @property
    def hop_samples(self):
        return int(self.sample_rate * self.hop_t + 0.5)

    @property
    def max_samples(self):
        return int(self.buffer_t * self.sample_rate)

    @property
    def feature_size(self):
        num_features = {
            Vectorizer.mfccs: self.n_mfcc,
            Vectorizer.mels: self.n_filt,
            Vectorizer.speechpy_mfccs: self.n_mfcc
        }[self.vectorizer]
        if self.use_delta:
            num_features *= 2
        return num_features

    def vectorization_md5_hash(self):
        """Hash all the fields related to audio vectorization"""
        keys = sorted(pr.__dict__)
        keys.remove('threshold_config')
        keys.remove('threshold_center')
        return hashlib.md5(
            str([pr.__dict__[i] for i in keys]).encode()
        ).hexdigest()


class Vectorizer:
    mels = 1
    mfccs = 2
    speechpy_mfccs = 3


# Global listener parameters
pr = ListenerParams(
    window_t=0.1, hop_t=0.05, buffer_t=1.5, sample_rate=16000,
    sample_depth=2, n_mfcc=13, n_filt=20, n_fft=512, use_delta=False,
    threshold_config=((6, 4),), threshold_center=0.2, vectorizer=Vectorizer.mfccs
)

# Used to fill in old param files without new attributes
compatibility_params = dict(vectorizer=Vectorizer.speechpy_mfccs)


def inject_params(model_name: str) -> ListenerParams:
    """Set the global listener params to a saved model"""
    params_file = model_name + '.params'
    try:
        with open(params_file) as f:
            pr.__dict__.update(compatibility_params, **json.load(f))
    except (OSError, ValueError, TypeError):
        if isfile(model_name):
            print('Warning: Failed to load parameters from ' + params_file)
    return pr


def save_params(model_name: str):
    """Save current global listener params to a file"""
    with open(model_name + '.params', 'w') as f:
        json.dump(pr.__dict__, f)
