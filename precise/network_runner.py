from importlib import import_module
from os.path import splitext
from typing import BinaryIO, Union

import numpy as np

from precise.common import buffer_to_audio, load_precise_model, inject_params


class TensorflowRunner:
    def __init__(self, model_name: str):
        if model_name.endswith('.net'):
            print('Warning: ', model_name, 'looks like a Keras model.')
        self.tf = import_module('tensorflow')
        self.graph = self.load_graph(model_name)

        self.inp_var = self.graph.get_operation_by_name('import/net_input').outputs[0]
        self.out_var = self.graph.get_operation_by_name('import/net_output').outputs[0]

        self.sess = self.tf.Session(graph=self.graph)

    def load_graph(self, model_file: str):  # returns: tf.Graph
        graph = self.tf.Graph()
        graph_def = self.tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            self.tf.import_graph_def(graph_def)

        return graph

    def run(self, inp: np.ndarray) -> float:
        return self.sess.run(self.out_var, {self.inp_var: inp[np.newaxis]})[0][0]


class KerasRunner:
    def __init__(self, model_name: str):
        import tensorflow as tf
        self.model = load_precise_model(model_name)
        self.graph = tf.get_default_graph()

    def run(self, inp: np.ndarray):
        with self.graph.as_default():
            return self.model.predict(np.array([inp]))[0][0]


class Listener:
    def __init__(self, model_name: str, chunk_size: int = -1, runner_cls: type = None):
        self.window_audio = np.array([])
        self.pr = inject_params(model_name)
        self.features = np.zeros((self.pr.n_features, self.pr.feature_size))
        self.read_size = -1 if chunk_size == -1 else self.pr.sample_depth * chunk_size
        runner_cls = runner_cls or self.find_runner(model_name)
        self.runner = runner_cls(model_name)
        self.mfcc = import_module('speechpy.feature').mfcc

    @staticmethod
    def find_runner(model_name):
        runners = {
            '.net': KerasRunner,
            '.pb': TensorflowRunner
        }
        ext = splitext(model_name)[-1]
        if ext not in runners:
            raise ValueError('File extension of ' + model_name + ' must be: ' + str(list(runners)))
        return runners[ext]

    def clear(self):
        self.window_audio = np.array([])
        self.features = np.zeros((self.pr.n_features, self.pr.feature_size))

    def update(self, stream: Union[BinaryIO, np.ndarray, bytes]) -> float:
        if isinstance(stream, np.ndarray):
            buffer_audio = stream
        else:
            if isinstance(stream, (bytes, bytearray)):
                chunk = stream
            else:
                chunk = stream.read(self.read_size)
            if len(chunk) == 0:
                raise EOFError
            buffer_audio = buffer_to_audio(chunk)

        self.window_audio = np.concatenate((self.window_audio, buffer_audio))

        if len(self.window_audio) >= self.pr.window_samples:
            new_features = self.mfcc(self.window_audio, self.pr.sample_rate, self.pr.window_t,
                                     self.pr.hop_t, self.pr.n_mfcc, self.pr.n_filt, self.pr.n_fft)
            self.window_audio = self.window_audio[len(new_features) * self.pr.hop_samples:]
            if len(new_features) > len(self.features):
                new_features = new_features[-len(self.features):]
            self.features = np.concatenate((self.features[len(new_features):], new_features))

        return self.runner.run(self.features)
