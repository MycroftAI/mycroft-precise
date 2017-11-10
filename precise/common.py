# Python 3
import numpy as np
from os.path import isfile
from precise.params import ListenerParams


pr = ListenerParams(window_t=0.1, hop_t=0.05, buffer_t=1.5,
                    sample_rate=16000, sample_depth=2,
                    n_mfcc=13, n_filt=20, n_fft=512)

lstm_units = 20


def vectorize_raw(audio):
    """Turns audio into feature vectors, without clipping for length"""
    from speechpy.main import mfcc
    return mfcc(audio, pr.sample_rate, pr.window_t, pr.hop_t, pr.n_mfcc, pr.n_filt, pr.n_fft)


def vectorize(audio):
    """
    Args:
        audio (array<float>): Audio verified to be of `sample_rate`

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


def load_vector(name, vectorizer=vectorize):
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


def load_audio(file):
    """
    Args:
        file (any): Audio filename or file object
    Returns:
        rate, array<float>: Sample rate and audio samples from 0..1
    """
    import wavio
    wav = wavio.read(file)
    if wav.data.dtype != np.int16:
        raise ValueError('Unsupported data type: ' + str(wav.data.dtype))
    if wav.rate != pr.sample_rate:
        raise ValueError('Unsupported sample rate: ' + str(wav.rate))

    data = np.squeeze(wav.data)
    return data.astype(np.float32) / float(np.iinfo(data.dtype).max)


def to_np(x):
    """list<np.array> to np.array"""
    arr = np.empty((len(x),) + x[0].shape)
    for i in range(len(x)):
        arr[i] = x[i]
    return arr


def glob_all(folder, filter):
    """Recursive glob"""
    import os
    import fnmatch
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, filter):
            matches.append(os.path.join(root, filename))
    return matches


def find_wavs(folder):
    """Finds keyword and not-keyword wavs in folder"""
    return glob_all(folder + '/keyword', '*.wav'), glob_all(folder + '/not-keyword', '*.wav')


def load_data(prefix):
    inputs = []
    outputs = []

    def add(filenames, output):
        nonlocal inputs, outputs
        inputs += [load_vector(f) for f in filenames]
        outputs += [np.array(output)] * len(filenames)

    kww, nkw = find_wavs(prefix)

    print('Loading keyword...')
    add(kww, 1.0)

    print('Loading not-keyword...')
    add(nkw, 0.0)

    return to_np(inputs), to_np(outputs)


def create_model(model_name, should_load):
    if isfile(model_name) and should_load:
        print('Loading from ' + model_name + '...')
        from keras.models import load_model
        model = load_model(model_name)
    else:
        from keras.layers.core import Dense
        from keras.layers.recurrent import GRU
        from keras.models import Sequential

        model = Sequential()
        model.add(GRU(lstm_units, activation='linear', input_shape=(pr.n_features, pr.feature_size), dropout=0.2, name='net'))
        model.add(Dense(1, activation='sigmoid'))

    model.compile('rmsprop', 'mse', metrics=['accuracy'])
    return model
