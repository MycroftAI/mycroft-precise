# Python 3

import numpy as np
from os import makedirs
from os.path import isfile, join, dirname
from typing import Tuple, List, Any
from precise.params import ListenerParams


pr = ListenerParams(window_t=0.1, hop_t=0.05, buffer_t=1.5,
                    sample_rate=16000, sample_depth=2,
                    n_mfcc=13, n_filt=20, n_fft=512)

lstm_units = 20
inhibit_t = 0.4
inhibit_dist_t = 1.0
inhibit_hop_t = 0.1


def vectorize_raw(audio: np.array) -> np.array:
    """Turns audio into feature vectors, without clipping for length"""
    from speechpy.main import mfcc
    return mfcc(audio, pr.sample_rate, pr.window_t, pr.hop_t, pr.n_mfcc, pr.n_filt, pr.n_fft)


def vectorize(audio: np.array) -> np.array:
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
        features = np.concatenate([np.zeros((pr.n_features - len(features), len(features[0]))), features])
    if len(features) > pr.n_features:
        features = features[-pr.n_features:]

    return features


def vectorize_inhibit(audio: np.array) -> np.array:
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


def load_vector(name: str, vectorizer=vectorize) -> np.array:
    """Loads and caches a vector input from a wav or npy file"""
    import os

    save_name = name if name.endswith('.npy') else os.path.join('cache', vectorizer.__name__ + name + '.npy')

    if os.path.isfile(save_name):
        return np.load(save_name)

    print('Loading ' + name + '...')
    os.makedirs(os.path.dirname(save_name), exist_ok=True)

    vec = vectorizer(load_audio(name))
    np.save(save_name, vec)
    return vec


def load_audio(file: Any) -> np.array:
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


def load_data(prefix: str) -> Tuple[np.array, np.array]:
    inputs = []
    outputs = []

    def add(filenames, output):
        nonlocal inputs, outputs
        inputs += [load_vector(f) for f in filenames]
        outputs += [np.array([output])] * len(filenames)

    kww, nkw = find_wavs(prefix)

    print('Loading keyword...')
    add(kww, 1.0)

    print('Loading not-keyword...')
    add(nkw, 0.0)

    return np.array(inputs), np.array(outputs)


def load_gen_data(prefix: str) -> Tuple[np.array, np.array]:
    """Generate inhibitory data"""
    kws, _ = find_wavs(prefix)
    inputs = np.empty((0, pr.n_features, pr.feature_size))
    for f in kws:
        new_vec = load_vector(f, vectorize_inhibit)
        inputs = np.concatenate([inputs, new_vec])
    return inputs, np.ones((len(inputs), 1))


def write_gen_data(prefix: str):
    """Copies fragments from keyword/ to not-keyword/generated/keyword/"""
    import wavio
    fd = join(prefix, 'not-keyword', 'generated')

    kws, _ = find_wavs(prefix)
    for f in kws:
        def samp(x):
            return int(pr.sample_rate * x)

        audio = load_audio(f)
        for offset in range(samp(inhibit_t), samp(inhibit_dist_t), samp(inhibit_hop_t)):
            if len(audio) - offset < samp(pr.buffer_t / 2.):
                break

            fn = join(fd, 'kw-' + f.replace(prefix + '/', ''))
            makedirs(dirname(fn), exist_ok=True)
            aud = audio[max(0, len(audio) - offset - samp(pr.buffer_t)):-offset]
            aud = (aud * np.iinfo(np.int16).max).astype(np.int16)
            wavio.write(fn, aud, pr.sample_rate)


def load_all_data(prefix: str) -> Tuple[np.array, np.array]:
    """Loads both data and generated inhibitory data"""
    inputs, outputs = load_data(prefix)
    gen_inputs, gen_outputs = load_gen_data(prefix)
    return np.concatenate([inputs, gen_inputs]), np.concatenate([outputs, gen_outputs])


def weighted_log_loss(yt, yp) -> Any:
    """Binary crossentropy with a bias towards false negatives"""
    from keras import backend as K

    weight = 0.5  # [0..1] where 1 is inf bias

    a = yt * K.log(yp + K.epsilon())
    b = (1 - yt) * K.log(1 + K.epsilon() - yp)
    return -1 * K.mean((1. - weight + yp) * (a + b))


def load_precise_model(model_name: str) -> Any:
    """Loads a Keras model from file, handling custom loss function"""
    import keras.losses
    keras.losses.weighted_log_loss = weighted_log_loss
    from keras.models import load_model
    return load_model(model_name)


def create_model(model_name: str, should_load: bool) -> Any:
    """
    Load or create a precise model

    Args:
        model_name: Name of model
        should_load: Whether to check if the model already exists

    Returns:
        model: Loaded Keras model
    """
    if isfile(model_name) and should_load:
        print('Loading from ' + model_name + '...')
        model = load_precise_model(model_name)
    else:
        from keras.layers.core import Dense
        from keras.layers.recurrent import GRU
        from keras.models import Sequential

        model = Sequential()
        model.add(GRU(lstm_units, activation='linear', input_shape=(pr.n_features, pr.feature_size), dropout=0.3, name='net'))
        model.add(Dense(1, activation='sigmoid'))

    model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
    return model
