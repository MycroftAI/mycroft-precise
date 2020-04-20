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
Train a model on infinitely generated batches

:model str
    Keras .net model file to load from and write to

:-e --epochs int 100
    Number of epochs to train on

:-b --batch-size int 200
    Number of samples in each batch

:-t --steps-per-epoch int 100
    Number of steps that are considered an epoch

:-c --chunk-size int 2048
    Number of audio samples between generating a training sample

:-r --random-data-folder str data/random
    Folder with properly encoded wav files of
    random audio that should not cause an activation

:-s --sensitivity float 0.2
    Target sensitivity when training. Higher values cause
    more false positives

:-sb --save-best
    Only save the model each epoch if its stats improve

:-nv --no-validation
    Disable accuracy and validation calculation
    to improve speed during training

:-mm --metric-monitor str loss
    Metric used to determine when to save

:-em --extra-metrics
    Add extra metrics during training

:-p --save-prob float 0.0
    Probability of saving audio into debug/ww and debug/nww folders

...
"""
from itertools import cycle
from math import sqrt

import numpy as np
from contextlib import suppress
from fitipy import Fitipy
from keras.callbacks import LambdaCallback
from os.path import splitext, join, basename
from prettyparse import Usage
from random import random, shuffle
from typing import *

from precise.model import create_model, ModelParams
from precise.network_runner import Listener
from precise.params import pr, save_params
from precise.scripts.base_script import BaseScript
from precise.train_data import TrainData
from precise.util import load_audio, glob_all, save_audio, chunk_audio


class TrainGeneratedScript(BaseScript):
    usage = Usage(__doc__) | TrainData.usage
    """A trainer the runs on generated data by overlaying wakewords on background audio"""

    def __init__(self, args):
        super().__init__(args)
        self.audio_buffer = np.zeros(pr.buffer_samples, dtype=float)
        self.vals_buffer = np.zeros(pr.buffer_samples, dtype=float)

        params = ModelParams(
            skip_acc=args.no_validation, extra_metrics=args.extra_metrics,
            loss_bias=1.0 - args.sensitivity
        )
        self.model = create_model(args.model, params)
        self.listener = Listener('', args.chunk_size, runner_cls=lambda x: None)

        from keras.callbacks import ModelCheckpoint, TensorBoard
        checkpoint = ModelCheckpoint(args.model, monitor=args.metric_monitor,
                                     save_best_only=args.save_best)
        epoch_fiti = Fitipy(splitext(args.model)[0] + '.epoch')
        self.epoch = epoch_fiti.read().read(0, int)

        def on_epoch_end(_a, _b):
            self.epoch += 1
            epoch_fiti.write().write(self.epoch, str)

        self.model_base = splitext(self.args.model)[0]

        self.callbacks = [
            checkpoint, TensorBoard(
                log_dir=self.model_base + '.logs',
            ), LambdaCallback(on_epoch_end=on_epoch_end)
        ]

        self.data = TrainData.from_both(args.tags_file, args.tags_folder, args.folder)
        pos_files, neg_files = self.data.train_files
        self.neg_files_it = iter(cycle(neg_files))
        self.pos_files_it = iter(cycle(pos_files))

    def layer_with(self, sample: np.ndarray, value: int) -> np.ndarray:
        """Create an identical 2d array where the second row is filled with value"""
        b = np.full((2, len(sample)), value, dtype=float)
        b[0] = sample
        return b

    def generate_wakeword_pieces(self, volume):
        """Generates chunks of audio that represent the wakeword stream"""
        while True:
            target = 1 if random() > 0.5 else 0
            it = self.pos_files_it if target else self.neg_files_it
            sample_file = next(it)
            yield self.layer_with(self.normalize_volume_to(load_audio(sample_file), volume), target)
            yield self.layer_with(np.zeros(int(pr.sample_rate * (0.5 + 2.0 * random()))), 0)

    def chunk_audio_pieces(self, pieces, chunk_size):
        """Convert chunks of audio into a series of equally sized pieces"""
        left_over = np.array([])
        for piece in pieces:
            if left_over.size == 0:
                combined = piece
            else:
                combined = np.concatenate([left_over, piece], axis=-1)
            for chunk in chunk_audio(combined.T, chunk_size):
                yield chunk.T
            left_over = piece[-(len(piece) % chunk_size):]

    def calc_volume(self, sample: np.ndarray):
        """Find the RMS of the audio"""
        return sqrt(np.mean(np.square(sample)))

    def normalize_volume_to(self, sample, volume):
        """Normalize the volume to a certain RMS"""
        return volume * sample / self.calc_volume(sample)

    def merge(self, a, b, ratio):
        """Perform a weighted sum of a and b. ratio=1.0 means 100% of b and 0% of a"""
        return (1.0 - ratio) * a + ratio * b

    @staticmethod
    def max_run_length(x: np.ndarray, val: int):
        """Finds the maximum continuous length of the given value in the sequence"""
        if x.size == 0:
            return 0
        else:
            y = np.array(x[1:] != x[:-1])
            i = np.append(np.where(y), len(x) - 1)
            run_lengths = np.diff(np.append(-1, i))
            run_length_values = x[i]
            return max([rl for rl, v in zip(run_lengths, run_length_values) if v == val], default=0)

    def vectors_from_fn(self, fn: str):
        """
        Run through a single background audio file, overlaying with wake words.
        Generates (mfccs, target) where mfccs is a series of mfcc values and
        target is a single integer classification of the target network output for that chunk
        """
        audio = load_audio(fn)
        audio_volume = self.calc_volume(audio)
        audio_volume *= 0.4 + 0.5 * random()
        audio = self.normalize_volume_to(audio, audio_volume)

        self.listener.clear()
        chunked_bg = chunk_audio(audio, self.args.chunk_size)
        chunked_ww = self.chunk_audio_pieces(self.generate_wakeword_pieces(audio_volume), self.args.chunk_size)

        for i, (chunk_bg, (chunk_ww, targets)) in enumerate(zip(chunked_bg, chunked_ww)):
            chunk = self.merge(chunk_bg, chunk_ww, 0.6)
            self.vals_buffer = np.concatenate((self.vals_buffer[len(targets):], targets))
            self.audio_buffer = np.concatenate((self.audio_buffer[len(chunk):], chunk))
            mfccs = self.listener.update_vectors(chunk)
            percent_overlapping = self.max_run_length(self.vals_buffer, 1) / len(self.vals_buffer)

            if self.vals_buffer[-1] == 0 and percent_overlapping > 0.8:
                target = 1
            elif percent_overlapping < 0.5:
                target = 0
            else:
                continue

            if random() > 1.0 - self.args.save_prob:
                name = splitext(basename(fn))[0]
                wav_file = join('debug', 'ww' if target == 1 else 'nww', '{} - {}.wav'.format(name, i))
                save_audio(wav_file, self.audio_buffer)
            yield mfccs, target

    @staticmethod
    def samples_to_batches(samples: Iterable, batch_size: int):
        """Chunk a series of network inputs and outputs into larger batches"""
        it = iter(samples)
        while True:
            with suppress(StopIteration):
                batch_in, batch_out = [], []
                for i in range(batch_size):
                    sample_in, sample_out = next(it)
                    batch_in.append(sample_in)
                    batch_out.append(sample_out)
            if not batch_in:
                raise StopIteration
            yield np.array(batch_in), np.array(batch_out)

    def generate_samples(self):
        """Generate training samples (network inputs and outputs)"""
        filenames = glob_all(self.args.random_data_folder, '*.wav')
        shuffle(filenames)
        while True:
            for fn in filenames:
                for x, y in self.vectors_from_fn(fn):
                    yield x, y

    def run(self):
        """Train the model on randomly generated batches"""
        _, test_data = self.data.load(train=False, test=True)
        try:
            self.model.fit_generator(
                self.samples_to_batches(self.generate_samples(), self.args.batch_size),
                steps_per_epoch=self.args.steps_per_epoch,
                epochs=self.epoch + self.args.epochs, validation_data=test_data,
                callbacks=self.callbacks, initial_epoch=self.epoch
            )
        finally:
            self.model.save(self.args.model)
            save_params(self.args.model)


main = TrainGeneratedScript.run_main

if __name__ == '__main__':
    main()
