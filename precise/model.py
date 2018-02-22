# Copyright (c) 2017 Mycroft AI Inc.
from os.path import isfile
from typing import *

from precise.functions import load_keras, false_pos, false_neg, weighted_log_loss
from precise.params import inject_params, pr

lstm_units = 20


def load_precise_model(model_name: str) -> Any:
    """Loads a Keras model from file, handling custom loss function"""
    if not model_name.endswith('.net'):
        print('Warning: Unknown model type, ', model_name)

    inject_params(model_name)
    return load_keras().models.load_model(model_name)


def create_model(model_name: str, skip_acc: bool = False) -> Any:
    """
    Load or create a precise model

    Args:
        model_name: Name of model
        skip_acc: Whether to skip accuracy calculation while training

    Returns:
        model: Loaded Keras model
    """
    if isfile(model_name):
        print('Loading from ' + model_name + '...')
        model = load_precise_model(model_name)
    else:
        from keras.layers.core import Dense
        from keras.layers.recurrent import GRU
        from keras.models import Sequential

        model = Sequential()
        model.add(GRU(lstm_units, activation='linear', input_shape=(pr.n_features, pr.feature_size),
                      dropout=0.3, name='net'))
        model.add(Dense(1, activation='sigmoid'))

    load_keras()
    metrics = ['accuracy', false_pos, false_neg]
    model.compile('rmsprop', weighted_log_loss, metrics=(not skip_acc) * metrics)
    return model
