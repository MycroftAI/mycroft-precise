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
Mathematical functions used to customize
computation in various places
"""
from math import exp, log, sqrt, pi
import numpy as np
from typing import *

LOSS_BIAS = 0.9  # [0..1] where 1 is inf bias


def set_loss_bias(bias: float):
    """
    Changes the loss bias

    This allows customizing the acceptable tolerance between
    false negatives and false positives

    Near 1.0 reduces false positives
    Near 0.0 reduces false negatives
    """
    global LOSS_BIAS
    LOSS_BIAS = bias


def weighted_log_loss(yt, yp) -> Any:
    """
    Binary crossentropy with a bias towards false negatives
    yt: Target
    yp: Prediction
    """
    from keras import backend as K

    pos_loss = -(0 + yt) * K.log(0 + yp + K.epsilon())
    neg_loss = -(1 - yt) * K.log(1 - yp + K.epsilon())

    return LOSS_BIAS * K.mean(neg_loss) + (1. - LOSS_BIAS) * K.mean(pos_loss)


def weighted_mse_loss(yt, yp) -> Any:
    """Standard mse loss with a weighting between false negatives and positives"""
    from keras import backend as K

    total = K.sum(K.ones_like(yt))
    neg_loss = total * K.sum(K.square(yp * (1 - yt))) / K.sum(1 - yt)
    pos_loss = total * K.sum(K.square(1. - (yp * yt))) / K.sum(yt)

    return LOSS_BIAS * neg_loss + (1. - LOSS_BIAS) * pos_loss


def false_pos(yt, yp) -> Any:
    """
    Metric for Keras that *estimates* false positives while training
    This will not be completely accurate because it weights batches
    equally
    """
    from keras import backend as K
    return K.sum(K.cast(yp * (1 - yt) > 0.5, 'float')) / K.maximum(1.0, K.sum(1 - yt))


def false_neg(yt, yp) -> Any:
    """
    Metric for Keras that *estimates* false negatives while training
    This will not be completely accurate because it weights batches
    equally
    """
    from keras import backend as K
    return K.sum(K.cast((1 - yp) * (0 + yt) > 0.5, 'float')) / K.maximum(1.0, K.sum(0 + yt))


def load_keras() -> Any:
    """Imports Keras injecting custom functions to prevent exceptions"""
    import keras
    keras.losses.weighted_log_loss = weighted_log_loss
    keras.metrics.false_pos = false_pos
    keras.metrics.false_positives = false_pos
    keras.metrics.false_neg = false_neg
    return keras


def sigmoid(x):
    """Sigmoid squashing function for scalars"""
    return 1 / (1 + exp(-x))


def asigmoid(x):
    """Inverse sigmoid (logit) for scalars"""
    return -log(1 / x - 1)


def pdf(x, mu, std):
    """Probability density function (normal distribution)"""
    if std == 0:
        return 0
    return (1.0 / (std * sqrt(2 * pi))) * np.exp(-(x - mu) ** 2 / (2 * std ** 2))
