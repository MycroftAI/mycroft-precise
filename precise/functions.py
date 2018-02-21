from typing import *


def weighted_log_loss(yt, yp) -> Any:
    """
    Binary crossentropy with a bias towards false negatives
    yt: Target
    yp: Prediction
    """
    from keras import backend as K
    weight = 0.99  # [0..1] where 1 is inf bias

    pos_loss = -(0 + yt) * K.log(0 + yp + K.epsilon())
    neg_loss = -(1 - yt) * K.log(1 - yp + K.epsilon())
    return weight * K.sum(neg_loss) + (1. - weight) * K.sum(pos_loss)


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
