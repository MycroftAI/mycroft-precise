# Python 3
# Copyright (c) 2017 Mycroft AI Inc.
import json
from os.path import isfile
from typing import Tuple, List, Any
from argparse import ArgumentParser

import numpy as np

from precise.params import ListenerParams

pr = ListenerParams(window_t=0.1, hop_t=0.05, buffer_t=1.5,
                    sample_rate=16000, sample_depth=2,
                    n_mfcc=13, n_filt=20, n_fft=512)

lstm_units = 20
inhibit_t = 0.4
inhibit_dist_t = 1.0
inhibit_hop_t = 0.1


def create_parser(usage: str) -> ArgumentParser:
    """
    Creates an argument parser from a condensed usage string in the format of:
    :pos_arg_name int
        This is the help message
        which can span multiple lines
    :-o --optional_arg str default_value
        The type can be any valid python type
    :-eo --extra-option
        This adds args.extra_option as a bool
        which is False by default
    """
    first_line = [i for i in usage.split('\n') if i][0]
    indent = ' ' * (len(first_line) - len(first_line.lstrip(' ')))
    usage = usage.replace('\n' + indent, '\n')

    defaults = {}
    description, *descriptors = usage.split('\n:')
    parser = ArgumentParser(description=description.strip())
    for descriptor in descriptors:
        try:
            options, *help = descriptor.split('\n')
            help = ' '.join(help).replace('    ', '')
            if options.count(' ') == 1:
                if options[0] == '-':
                    short, long = options.split(' ')
                    var_name = long.strip('-').replace('-', '_')
                    parser.add_argument(short, long, dest=var_name, action='store_true', help=help)
                    defaults[var_name] = False
                else:
                    short, typ = options.split(' ')
                    parser.add_argument(short, type=eval(typ), help=help)
            else:
                short, long, typ, default = options.split(' ')
                help += '. Default: ' + default
                default = '' if default == '-' else default
                parser.add_argument(short, long, type=eval(typ), default=default, help=help)
        except Exception as e:
            print(e.__class__.__name__ + ': ' + str(e))
            print('While parsing:')
            print(descriptor)
            exit(1)
    return parser


def buffer_to_audio(buffer: bytes) -> np.ndarray:
    """Convert a raw mono audio byte string to numpy array of floats"""
    return np.fromstring(buffer, dtype='<i2').astype(np.float32, order='C') / 32768.0


def inject_params(model_name: str) -> ListenerParams:
    params_file = model_name + '.params'
    try:
        global pr
        with open(params_file) as f:
            pr = ListenerParams(**json.load(f))
    except (OSError, ValueError, TypeError):
        print('Warning: Failed to load parameters from ' + params_file)
    return pr


def save_params(model_name: str):
    with open(model_name + '.params', 'w') as f:
        json.dump(pr._asdict(), f)


def vectorize_raw(audio: np.ndarray) -> np.ndarray:
    """Turns audio into feature vectors, without clipping for length"""
    from speechpy.feature import mfcc
    return mfcc(audio, pr.sample_rate, pr.window_t, pr.hop_t, pr.n_mfcc, pr.n_filt, pr.n_fft)


def vectorize(audio: np.ndarray) -> np.ndarray:
    """
    Args:
        audio: Audio verified to be of `sample_rate`

    Returns:
        array<float>: Vector representation of audio
    """
    if len(audio) > pr.max_samples:
        audio = audio[-pr.max_samples:]
    features = vectorize_raw(audio)
    if len(features) < pr.n_features:
        features = np.concatenate(
            [np.zeros((pr.n_features - len(features), len(features[0]))), features])
    if len(features) > pr.n_features:
        features = features[-pr.n_features:]

    return features


def vectorize_inhibit(audio: np.ndarray) -> np.ndarray:
    """
    Returns an array of inputs generated from the
    keyword audio that shouldn't cause an activation
    """

    def samp(x):
        return int(pr.sample_rate * x)

    inputs = []
    for offset in range(samp(inhibit_t), samp(inhibit_dist_t), samp(inhibit_hop_t)):
        if len(audio) - offset < samp(pr.buffer_t / 2.):
            break
        inputs.append(vectorize(audio[:-offset]))
    return np.array(inputs) if inputs else np.empty((0, pr.n_features, pr.feature_size))


def load_vector(name: str, vectorizer=vectorize) -> np.ndarray:
    """Loads and caches a vector input from a wav or npy file"""
    import os

    save_name = name if name.endswith('.npy') else os.path.join('cache', str(abs(hash(pr))),
                                                                vectorizer.__name__ + '.' + name + '.npy')

    if os.path.isfile(save_name):
        return np.load(save_name)

    print('Loading ' + name + '...')
    os.makedirs(os.path.dirname(save_name), exist_ok=True)

    vec = vectorizer(load_audio(name))
    np.save(save_name, vec)
    return vec


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


def glob_all(folder: str, filter: str) -> List[str]:
    """Recursive glob"""
    import os
    import fnmatch
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, filter):
            matches.append(os.path.join(root, filename))
    return matches


def find_wavs(folder: str) -> Tuple[List[str], List[str]]:
    """Finds keyword and not-keyword wavs in folder"""
    return glob_all(folder + '/keyword', '*.wav'), glob_all(folder + '/not-keyword', '*.wav')


def weighted_log_loss(yt, yp) -> Any:
    """
    Binary crossentropy with a bias towards false negatives
    yt: Target
    yp: Prediction
    """
    from keras import backend as K
    weight = 0.9  # [0..1] where 1 is inf bias

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


def false_pos(yt, yp):
    from keras import backend as K
    return K.sum(K.cast(yp * (1 - yt) > 0.5, 'float')) / K.sum(1 - yt)


def false_neg(yt, yp):
    from keras import backend as K
    return K.sum(K.cast((1 - yp) * (0 + yt) > 0.5, 'float')) / K.sum(0 + yt)


def load_keras():
    import keras
    keras.losses.weighted_log_loss = weighted_log_loss
    keras.metrics.false_pos = false_pos
    keras.metrics.false_neg = false_neg
    return keras


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
