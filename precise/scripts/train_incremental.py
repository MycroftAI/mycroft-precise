#!/usr/bin/env python3
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
import numpy as np
from os import makedirs
from os.path import basename, splitext, isfile, join
from prettyparse import create_parser
from random import random
from typing import *

from precise.model import create_model, ModelParams
from precise.network_runner import Listener, KerasRunner
from precise.params import pr
from precise.train_data import TrainData
from precise.trainer import Trainer
from precise.util import load_audio, save_audio, glob_all

usage = '''
    Train a model to inhibit activation by
    marking false activations and retraining
    
    :-e --epochs int 1
        Number of epochs to train before continuing evaluation
    
    :-ds --delay-samples int 10
        Number of false activations to save before re-training
    
    :-c --chunk-size int 2048
        Number of samples between testing the neural network
    
    :-r --random-data-folder str data/random
        Folder with properly encoded wav files of
        random audio that should not cause an activation
    
    ...
'''


def chunk_audio(audio: np.ndarray, chunk_size: int) -> Generator[np.ndarray, None, None]:
    for i in range(chunk_size, len(audio), chunk_size):
        yield audio[i - chunk_size:i]


def load_trained_fns(model_name: str) -> list:
    progress_file = model_name.replace('.net', '') + '.trained.txt'
    if isfile(progress_file):
        print('Starting from saved position in', progress_file)
        with open(progress_file, 'rb') as f:
            return f.read().decode('utf8', 'surrogatepass').split('\n')
    return []


def save_trained_fns(trained_fns: list, model_name: str):
    with open(model_name.replace('.net', '') + '.trained.txt', 'wb') as f:
        f.write('\n'.join(trained_fns).encode('utf8', 'surrogatepass'))


class IncrementalTrainer(Trainer):
    def __init__(self):
        super().__init__(create_parser(usage))

        for i in (
                join(self.args.folder, 'not-wake-word', 'generated'),
                join(self.args.folder, 'test', 'not-wake-word', 'generated')
        ):
            makedirs(i, exist_ok=True)

        self.trained_fns = load_trained_fns(self.args.model)
        self.audio_buffer = np.zeros(pr.buffer_samples, dtype=float)

        if not isfile(self.args.model):
            params = ModelParams(
                skip_acc=self.args.no_validation, extra_metrics=self.args.extra_metrics
            )
            create_model(self.args.model, params).save(self.args.model)
        self.listener = Listener(self.args.model, self.args.chunk_size, runner_cls=KerasRunner)

    @staticmethod
    def load_data(args: Any):
        data = TrainData.from_tags(args.tags_file, args.tags_folder)
        return data.load(True, not args.no_validation)

    def retrain(self):
        """Train for a session, pulling in any new data from the filesystem"""
        folder = TrainData.from_folder(self.args.folder)
        train_data, test_data = folder.load(True, not self.args.no_validation)

        train_data = TrainData.merge(train_data, self.sampled_data)
        test_data = TrainData.merge(test_data, self.test)
        print()
        try:
            self.listener.runner.model.fit(
                *train_data, self.args.batch_size, self.epoch + self.args.epochs,
                validation_data=test_data, callbacks=self.callbacks,
                initial_epoch=self.epoch
            )
        finally:
            self.listener.runner.model.save(self.args.model)

    def train_on_audio(self, fn: str):
        """Run through a single audio file"""
        save_test = random() > 0.8
        samples_since_train = 0
        audio = load_audio(fn)
        num_chunks = len(audio) // self.args.chunk_size

        self.listener.clear()

        for i, chunk in enumerate(chunk_audio(audio, self.args.chunk_size)):
            print('\r' + str(i * 100. / num_chunks) + '%', end='', flush=True)
            self.audio_buffer = np.concatenate((self.audio_buffer[len(chunk):], chunk))
            conf = self.listener.update(chunk)
            if conf > 0.5:
                samples_since_train += 1
                name = splitext(basename(fn))[0] + '-' + str(i) + '.wav'
                name = join(self.args.folder, 'test' if save_test else '', 'not-wake-word',
                            'generated', name)
                save_audio(name, self.audio_buffer)
                print()
                print('Saved to:', name)

            if not save_test and samples_since_train >= self.args.delay_samples and self.args.epochs > 0:
                samples_since_train = 0
                self.retrain()

    def run(self):
        """
        Begin reading through audio files, saving false
        activations and retraining when necessary
        """
        for fn in glob_all(self.args.random_data_folder, '*.wav'):
            if fn in self.trained_fns:
                print('Skipping ' + fn + '...')
                continue

            print('Starting file ' + fn + '...')
            self.train_on_audio(fn)
            print('\r100%                 ')

            self.trained_fns.append(fn)
            save_trained_fns(self.trained_fns, self.args.model)


def main():
    try:
        IncrementalTrainer().run()
    except KeyboardInterrupt:
        print()


if __name__ == '__main__':
    main()
