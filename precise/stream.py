#!/usr/bin/env python3

import sys
sys.path += ['.']  # noqa

import os
import json
import numpy as np
from json.decoder import JSONDecodeError
from speechpy.main import mfcc
from precise.params import ListenerParams


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def buffer_to_audio(buffer):
    """Convert a raw mono audio byte string to numpy array of floats"""
    return np.fromstring(buffer, dtype='<i2').astype(np.float32, order='C') / 32768.0


class NetworkRunner:
    def __init__(self, model_name):
        self.graph = load_graph(model_name)

        self.inp_var = self.graph.get_operation_by_name('import/net_input').outputs[0]
        self.out_var = self.graph.get_operation_by_name('import/net_output').outputs[0]

        self.sess = tf.Session(graph=self.graph)

    def run(self, inp):
        return self.sess.run(self.out_var, {self.inp_var: inp[np.newaxis]})[0][0]


class Listener:
    def __init__(self, model_name, chunk_size):
        self.buffer = np.array([])
        self.pr = self._load_params(model_name)
        self.features = np.zeros((self.pr.n_features, self.pr.feature_size))
        self.read_size = -1 if chunk_size == -1 else self.pr.sample_depth * chunk_size
        self.runner = NetworkRunner(model_name)

    def _load_params(self, model_name):
        try:
            with open(model_name + '.params') as f:
                return ListenerParams(**json.loads(f))
        except (OSError, JSONDecodeError, TypeError):
            from precise.common import pr
            return pr

    def update(self, stream):
        chunk = stream.read(self.read_size)
        if len(chunk) == 0:
            raise EOFError

        chunk_audio = buffer_to_audio(chunk)
        self.buffer = np.concatenate((self.buffer, chunk_audio))

        if len(self.buffer) >= self.pr.window_samples:
            remaining = self.pr.window_samples - (
                self.pr.hop_samples - (len(self.buffer) - self.pr.window_samples) % self.pr.hop_samples)
            new_features = mfcc(self.buffer, self.pr.sample_rate, self.pr.window_t, self.pr.hop_t, self.pr.n_mfcc, self.pr.n_filt, self.pr.n_fft)

            self.features = np.concatenate([self.features[len(new_features):], new_features])
            self.buffer = self.buffer[-remaining:]

        return self.runner.run(self.features)


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    stdout = sys.stdout
    sys.stdout = sys.stderr

    if sys.stdin.isatty() or len(sys.argv) > 3 or len(sys.argv) == 1 or (len(sys.argv) == 3 and not sys.argv[2].isdigit()):
        print('Usage:', sys.argv[0], 'MODEL_NAME [CHUNK_SIZE] < audio.wav')
        print('    stdin should be a stream of raw int16 audio,')
        print('    written in groups of CHUNK_SIZE samples.')
        print()
        print('    If no CHUNK_SIZE is given it will read until EOF.')
        print()
        print('    For every chunk, an inference will be given')
        print('    via stdout as a float string, one per line')
        sys.exit(1)

    global tf
    import tensorflow
    tf = tensorflow

    listener = Listener(sys.argv[1], int(sys.argv[2]) if 2 < len(sys.argv) else -1)

    try:
        while True:
            conf = listener.update(sys.stdin.buffer)
            stdout.buffer.write((str(conf) + '\n').encode('ascii'))
            stdout.buffer.flush()
    except (EOFError, KeyboardInterrupt):
        pass

if __name__ == '__main__':
    main()
