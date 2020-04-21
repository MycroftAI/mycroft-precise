#!/usr/bin/env python3
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
Simulate listening to long chunks of audio to find
unbiased false positive metrics

:model str
    Either Keras (.net) or TensorFlow (.pb) model to test

:folder str
    Folder with a set of long wav files to test against

:-c --chunk_size int 4096
    Number of samples between tests

:-t --threshold float 0.5
    Network output required to be considered an activation
"""
import attr
import numpy as np
from glob import glob
from os.path import join, basename
from precise_runner.runner import TriggerDetector
from prettyparse import Usage

from precise.network_runner import Listener
from precise.params import pr, inject_params
from precise.scripts.base_script import BaseScript
from precise.util import load_audio
from precise.vectorization import vectorize_raw


@attr.s()
class Metric:
    chunk_size = attr.ib()  # type: int
    seconds = attr.ib(0.0)  # type: float
    activated_chunks = attr.ib(0)  # type: int
    activations = attr.ib(0)  # type: int
    activation_sum = attr.ib(0.0)  # type: float

    @property
    def days(self):
        return self.seconds / (60 * 60 * 24)

    def add(self, other):
        self.seconds += other.seconds
        self.activated_chunks += other.activated_chunks
        self.activations += other.activations
        self.activation_sum += other.activation_sum

    @property
    def chunks(self):
        return self.seconds * pr.sample_rate / self.chunk_size

    def info_string(self, title):
        return (
            '=== {title} ===\n'
            'Hours: {hours:.2f}\n'
            'Activations / Day: {activations_per_day:.2f}\n'
            'Activated Chunks / Day: {chunks_per_day:.2f}\n'
            'Average Activation (*100): {average_activation:.2f}'.format(
                title=title,
                hours=self.days * 24,
                activations_per_day=self.activations / self.days,
                chunks_per_day=self.activated_chunks / self.days,
                average_activation=100.0 * self.activation_sum / self.chunks
            )
        )


class SimulateScript(BaseScript):
    usage = Usage(__doc__)

    def __init__(self, args):
        super().__init__(args)
        inject_params(self.args.model)
        self.runner = Listener.find_runner(self.args.model)(self.args.model)
        self.audio_buffer = np.zeros(pr.buffer_samples, dtype=float)

    def evaluate(self, audio: np.ndarray) -> np.ndarray:
        print('MFCCs...')
        mfccs = vectorize_raw(audio)
        print('Splitting...')
        mfcc_hops = self.args.chunk_size // pr.hop_samples
        inputs = np.array([
            mfccs[i - pr.n_features:i] for i in range(pr.n_features, len(mfccs), mfcc_hops)
        ])
        del mfccs
        print('Predicting...')
        predictions = self.runner.predict(inputs)
        del inputs
        return predictions

    def run(self):
        total = Metric(chunk_size=self.args.chunk_size)
        for i in glob(join(self.args.folder, '*.wav')):
            audio = load_audio(i)
            if audio.size == 0:
                continue

            predictions = self.evaluate(audio)
            detector = TriggerDetector(self.args.chunk_size, trigger_level=0, sensitivity=self.args.threshold)

            metric = Metric(
                chunk_size=self.args.chunk_size,
                seconds=len(audio) / pr.sample_rate,
                activated_chunks=(predictions > detector.sensitivity).sum(),
                activations=sum(detector.update(i) for i in predictions),
                activation_sum=predictions.sum()
            )
            total.add(metric)
            print()
            print(metric.info_string(basename(i)))
            del audio
        print()
        print()
        print(total.info_string('Total'))


main = SimulateScript.run_main

if __name__ == '__main__':
    main()
