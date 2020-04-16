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
Pieces that convert audio to predictions
"""
import numpy as np
from abc import abstractmethod, ABCMeta
from importlib import import_module
from os.path import splitext
from typing import *
from typing import BinaryIO

from precise.threshold_decoder import ThresholdDecoder
from precise.model import load_precise_model
from precise.params import inject_params, pr
from precise.util import buffer_to_audio
from precise.vectorization import vectorize_raw, add_deltas


class Runner(metaclass=ABCMeta):
    """
    Classes that execute trained models on vectorized audio
    and produce prediction values
    """
    @abstractmethod
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def run(self, inp: np.ndarray) -> float:
        pass


class TensorFlowRunner(Runner):
    """Executes a frozen Tensorflow model created from precise-convert"""
    def __init__(self, model_name: str):
        if model_name.endswith('.net'):
            print('Warning: ', model_name, 'looks like a Keras model.')
        self.tf = import_module('tensorflow')
        self.graph = self.load_graph(model_name)

        self.inp_var = self.graph.get_operation_by_name('import/net_input').outputs[0]
        self.out_var = self.graph.get_operation_by_name('import/net_output').outputs[0]

        self.sess = self.tf.Session(graph=self.graph)

    def load_graph(self, model_file: str) -> 'tf.Graph':
        graph = self.tf.Graph()
        graph_def = self.tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            self.tf.import_graph_def(graph_def)

        return graph

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run on multiple inputs"""
        return self.sess.run(self.out_var, {self.inp_var: inputs})

    def run(self, inp: np.ndarray) -> float:
        return self.predict(inp[np.newaxis])[0][0]


class KerasRunner(Runner):
    """ Executes a regular Keras model created from precise-train"""
    def __init__(self, model_name: str):
        import tensorflow as tf
        # ISSUE 88 - Following 3 lines added to resolve issue 88 - JM 2020-02-04 per liny90626
        from tensorflow.python.keras.backend import set_session # ISSUE 88
        self.sess = tf.Session() # ISSUE 88
        set_session(self.sess) # ISSUE 88
        self.model = load_precise_model(model_name)
        self.graph = tf.get_default_graph()

    def predict(self, inputs: np.ndarray):
        from tensorflow.python.keras.backend import set_session		# ISSUE 88
        with self.graph.as_default():
            set_session(self.sess)		# ISSUE 88
            return self.model.predict(inputs)

    def run(self, inp: np.ndarray) -> float:
        return self.predict(inp[np.newaxis])[0][0]


class Listener:
    """Listener that preprocesses audio into MFCC vectors and executes neural networks"""

    def __init__(self, model_name: str, chunk_size: int = -1, runner_cls: type = None):
        self.window_audio = np.array([])
        self.pr = inject_params(model_name)
        self.mfccs = np.zeros((self.pr.n_features, self.pr.n_mfcc))
        self.chunk_size = chunk_size
        runner_cls = runner_cls or self.find_runner(model_name)
        self.runner = runner_cls(model_name)
        self.threshold_decoder = ThresholdDecoder(self.pr.threshold_config, pr.threshold_center)

    @staticmethod
    def find_runner(model_name: str) -> Type[Runner]:
        runners = {
            '.net': KerasRunner,
            '.pb': TensorFlowRunner
        }
        ext = splitext(model_name)[-1]
        if ext not in runners:
            raise ValueError('File extension of ' + model_name + ' must be: ' + str(list(runners)))
        return runners[ext]

    def clear(self):
        self.window_audio = np.array([])
        self.mfccs = np.zeros((self.pr.n_features, self.pr.n_mfcc))

    def update_vectors(self, stream: Union[BinaryIO, np.ndarray, bytes]) -> np.ndarray:
        if isinstance(stream, np.ndarray):
            buffer_audio = stream
        else:
            if isinstance(stream, (bytes, bytearray)):
                chunk = stream
            else:
                chunk = stream.read(self.chunk_size)
            if len(chunk) == 0:
                raise EOFError
            buffer_audio = buffer_to_audio(chunk)

        self.window_audio = np.concatenate((self.window_audio, buffer_audio))

        if len(self.window_audio) >= self.pr.window_samples:
            new_features = vectorize_raw(self.window_audio)
            self.window_audio = self.window_audio[len(new_features) * self.pr.hop_samples:]
            if len(new_features) > len(self.mfccs):
                new_features = new_features[-len(self.mfccs):]
            self.mfccs = np.concatenate((self.mfccs[len(new_features):], new_features))

        return self.mfccs

    def update(self, stream: Union[BinaryIO, np.ndarray, bytes]) -> float:
        mfccs = self.update_vectors(stream)
        if self.pr.use_delta:
            mfccs = add_deltas(mfccs)
        raw_output = self.runner.run(mfccs)
        return self.threshold_decoder.decode(raw_output)
