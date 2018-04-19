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


def weighted_log_loss(yt, yp) -> Any:
    """
    Binary crossentropy with a bias towards false negatives
    yt: Target
    yp: Prediction
    """
    from keras import backend as K
    weight = 0.7  # [0..1] where 1 is inf bias

    pos_loss = -(0 + yt) * K.log(0 + yp + K.epsilon())
    neg_loss = -(1 - yt) * K.log(1 - yp + K.epsilon())

    return weight * K.mean(neg_loss) + (1. - weight) * K.mean(pos_loss)


def weighted_mse_loss(yt, yp) -> Any:
    from keras import backend as K
    weight = 0.9  # [0..1] where 1 is inf bias

    total = K.sum(K.ones_like(yt))
    neg_loss = total * K.sum(K.square(yp * (1 - yt))) / K.sum(1 - yt)
    pos_loss = total * K.sum(K.square(1. - (yp * yt))) / K.sum(yt)

    return weight * neg_loss + (1. - weight) * pos_loss


def false_pos(yt, yp) -> Any:
    from keras import backend as K
    return K.sum(K.cast(yp * (1 - yt) > 0.5, 'float')) / K.sum(1 - yt)


def false_neg(yt, yp) -> Any:
    from keras import backend as K
    return K.sum(K.cast((1 - yp) * (0 + yt) > 0.5, 'float')) / K.sum(0 + yt)


def load_keras() -> Any:
    import keras
    keras.losses.weighted_log_loss = weighted_log_loss
    keras.metrics.false_pos = false_pos
    keras.metrics.false_positives = false_pos
    keras.metrics.false_neg = false_neg
    return keras
