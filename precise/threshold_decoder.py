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
Code for converting network output to confidence level
"""
import numpy as np
from typing import Tuple

from precise.functions import asigmoid, sigmoid, pdf


class ThresholdDecoder:
    """
    Decode raw network output into a relatively linear threshold using
    This works by estimating the logit normal distribution of network
    activations using a series of averages and standard deviations to
    calculate a cumulative probability distribution

    Background:
    We could simply take the output of the neural network as the confidence of a given
    prediction, but this typically jumps quickly between 0.01 and 0.99 even in cases where
    the network is less confident about a prediction. This is a symptom of the sigmoid squashing
    high values to values close to 1. This ThresholdDecoder measures the average output of
    the network over a dataset and uses that to create a smooth distribution so that an output
    of 80% means that the network output is greater than roughly 80% of the dataset
    """
    def __init__(self, mu_stds: Tuple[Tuple[float, float]], center=0.5, resolution=200, min_z=-4, max_z=4):
        self.min_out = int(min(mu + min_z * std for mu, std in mu_stds))
        self.max_out = int(max(mu + max_z * std for mu, std in mu_stds))
        self.out_range = self.max_out - self.min_out
        self.cd = np.cumsum(self._calc_pd(mu_stds, resolution))
        self.center = center

    def decode(self, raw_output: float) -> float:
        if raw_output == 1.0 or raw_output == 0.0:
            return raw_output
        if self.out_range == 0:
            cp = int(raw_output > self.min_out)
        else:
            ratio = (asigmoid(raw_output) - self.min_out) / self.out_range
            ratio = min(max(ratio, 0.0), 1.0)
            cp = self.cd[int(ratio * (len(self.cd) - 1) + 0.5)]
        if cp < self.center:
            return 0.5 * cp / self.center
        else:
            return 0.5 + 0.5 * (cp - self.center) / (1 - self.center)

    def encode(self, threshold: float) -> float:
        threshold = 0.5 * threshold / self.center
        if threshold < 0.5:
            cp = threshold * self.center * 2
        else:
            cp = (threshold - 0.5) * 2 * (1 - self.center) + self.center
        ratio = np.searchsorted(self.cd, cp) / len(self.cd)
        return sigmoid(self.min_out + self.out_range * ratio)

    def _calc_pd(self, mu_stds, resolution):
        points = np.linspace(self.min_out, self.max_out, resolution * self.out_range)
        return np.sum([pdf(points, mu, std) for mu, std in mu_stds], axis=0) / (resolution * len(mu_stds))
