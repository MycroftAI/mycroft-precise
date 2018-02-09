#!/usr/bin/env python3
# Copyright (c) 2017 Mycroft AI Inc.
# This script trains the network, selectively choosing
# segments from data/random that cause an activation. These
# segments are moved into data/not-keyword and the network is retrained

import sys

sys.path += ['.']  # noqa

import numpy as np
from os import makedirs
from random import random
from glob import glob
from os.path import basename, splitext, isfile, join

from precise.train_data import TrainData
from precise.network_runner import Listener, KerasRunner
from precise.common import create_model, load_audio, save_audio, inject_params, create_parser

usage = """
Train a model to inhibit activation by
marking false activations and retraining

:model str
    Keras <NAME>.net file to train

:-e --epochs int 1
    Number of epochs to train before continuing evaluation

:-ds --delay-samples int 10
    Number of timesteps of false activations to save before re-training

:-c --chunk-size int 2048
    Number of samples between testing the neural network

:-b --batch-size int 128
    Batch size used for training

:-sb --save-best
    Only save the model each epoch if its stats improve

:-mm --metric-monitor str loss
    Metric used to determine when to save

:-nv --no-validation
    Disable accuracy and validation calculation
    to improve speed during training

:-r --random-data-dir str data/random
    Directories with properly encoded wav files of
    random audio that should not cause an activation
"""


def chunk_audio(audio: np.array, chunk_size: int):
    for i in range(chunk_size, len(audio), chunk_size):
        yield audio[i - chunk_size:i]


def load_trained_fns(model_name):
    progress_file = model_name.replace('.net', '') + '.trained.txt'
    if isfile(progress_file):
        print('Starting from saved position in', progress_file)
        with open(progress_file, 'rb') as f:
            return f.read().decode('utf8', 'surrogatepass').split('\n')
    return []


def save_trained_fns(trained_fns, model_name):
    with open(model_name.replace('.net', '') + '.trained.txt', 'wb') as f:
        f.write('\n'.join(trained_fns).encode('utf8', 'surrogatepass'))


class IncrementalTrainer:
    def __init__(self, args):
        self.args = args
        self.trained_fns = load_trained_fns(args.model)
        pr = inject_params(args.model)
        self.audio_buffer = np.zeros(pr.buffer_samples, dtype=float)

        from keras.callbacks import ModelCheckpoint
        self.checkpoint = ModelCheckpoint(args.model, monitor=args.metric_monitor,
                                          save_best_only=args.save_best)
        self.db_data = TrainData.from_db(args.db_file, args.db_folder).load(args.no_validation)

        if not isfile(args.model):
            create_model(args.model, args.no_validation).save(args.model)
        self.listener = Listener(args.model, args.chunk_size, runner_cls=KerasRunner)

    def retrain(self):
        """Train for a session, pulling in any new data from the filesystem"""
        folder = TrainData.from_folder(self.args.data_dir)
        train_data, test_data = folder.load(self.args.no_validation)

        train_data = TrainData.merge(train_data, self.db_data[0])
        test_data = TrainData.merge(test_data, self.db_data[1])
        print()
        try:
            self.listener.runner.model.fit(*train_data, self.args.batch_size, self.args.epochs,
                                           validation_data=test_data, callbacks=[self.checkpoint])
        finally:
            self.listener.runner.model.save(self.args.model)

    def train_on_audio(self, fn: str):
        """Run through a single audio file"""
        save_test = False
        samples_since_train = 0
        audio = load_audio(fn)
        num_chunks = len(audio) // self.args.chunk_size

        self.listener.clear()

        for i, chunk in enumerate(chunk_audio(audio, self.args.chunk_size)):
            print('\r' + str(i * 100. / num_chunks) + '%', end='', flush=True)
            audio_buffer = np.concatenate((self.audio_buffer[len(chunk):], chunk))
            conf = self.listener.update(chunk)
            if conf > 0.5:
                samples_since_train += 1
                name = splitext(basename(fn))[0] + '-' + str(i) + '.wav'
                name = join(self.args.data_dir, 'test' if save_test else '', 'not-keyword',
                            'generated', name)
                save_audio(name, audio_buffer)
                print()
                print('Saved to:', name)
            elif samples_since_train > 0:
                samples_since_train = self.args.delay_samples
            else:
                save_test = random() > 0.8

            if samples_since_train >= self.args.delay_samples and self.args.epochs > 0:
                samples_since_train = 0
                self.retrain()

    def train_incremental(self):
        """
        Begin reading through audio files, saving false
        activations and retraining when necessary
        """
        for fn in glob(self.args.random_data_dir + '/*.wav'):
            if fn in self.trained_fns:
                print('Skipping ' + fn + '...')
                continue

            print('Starting file ' + fn + '...')
            self.train_on_audio(fn)
            print('\r100%                 ')

            self.trained_fns.append(fn)
            save_trained_fns(self.trained_fns, self.args.model)


def main():
    args = TrainData.parse_args(create_parser(usage))

    for i in (
            join(args.data_dir, 'not-keyword', 'generated'),
            join(args.data_dir, 'test', 'not-keyword', 'generated')
    ):
        makedirs(i, exist_ok=True)

    trainer = IncrementalTrainer(args)
    try:
        trainer.train_incremental()
    except KeyboardInterrupt:
        print()


if __name__ == '__main__':
    main()
