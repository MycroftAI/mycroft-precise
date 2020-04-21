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
Train a new model on a dataset

:model str
    Keras model file (.net) to load from and save to

:-sf --samples-file str -
    Loads subset of data from the provided json file
    generated with precise-train-sampled

:-is --invert-samples
    Loads subset of data not inside --samples-file

:-e --epochs int 10
    Number of epochs to train model for

:-s --sensitivity float 0.2
    Target sensitivity when training. Higher values cause
    more false positives

:-b --batch-size int 5000
    Batch size for training

:-sb --save-best
    Only save the model each epoch if its stats improve

:-nv --no-validation
    Disable accuracy and validation calculation
    to improve speed during training

:-mm --metric-monitor str loss
    Metric used to determine when to save

:-em --extra-metrics
    Add extra metrics during training

:-f --freeze-till int 0
    Freeze all weights up to this index (non-inclusive).
    Can be negative to wrap from end

...
"""
from fitipy import Fitipy
from keras.callbacks import LambdaCallback
from os.path import splitext, isfile
from prettyparse import Usage
from typing import Any, Tuple

from precise.model import create_model, ModelParams
from precise.params import inject_params, save_params
from precise.scripts.base_script import BaseScript
from precise.train_data import TrainData
from precise.util import calc_sample_hash


class TrainScript(BaseScript):
    usage = Usage(__doc__) | TrainData.usage

    def __init__(self, args):
        super().__init__(args)

        if args.invert_samples and not args.samples_file:
            raise ValueError('You must specify --samples-file when using --invert-samples')
        if args.samples_file and not isfile(args.samples_file):
            raise ValueError('No such file: ' + (args.invert_samples or args.samples_file))
        if not 0.0 <= args.sensitivity <= 1.0:
            raise ValueError('sensitivity must be between 0.0 and 1.0')

        inject_params(args.model)
        save_params(args.model)
        params = ModelParams(skip_acc=args.no_validation, extra_metrics=args.extra_metrics,
                             loss_bias=1.0 - args.sensitivity, freeze_till=args.freeze_till)
        self.model = create_model(args.model, params)
        self.train, self.test = self.load_data(self.args)

        from keras.callbacks import ModelCheckpoint, TensorBoard
        checkpoint = ModelCheckpoint(args.model, monitor=args.metric_monitor,
                                     save_best_only=args.save_best)
        epoch_fiti = Fitipy(splitext(args.model)[0] + '.epoch')
        self.epoch = epoch_fiti.read().read(0, int)

        def on_epoch_end(_a, _b):
            self.epoch += 1
            epoch_fiti.write().write(self.epoch, str)

        self.model_base = splitext(self.args.model)[0]

        if args.samples_file:
            self.samples, self.hash_to_ind = self.load_sample_data(args.samples_file, self.train)
        else:
            self.samples = set()
            self.hash_to_ind = {}

        self.callbacks = [
            checkpoint, TensorBoard(
                log_dir=self.model_base + '.logs',
            ), LambdaCallback(on_epoch_end=on_epoch_end)
        ]

    @staticmethod
    def load_sample_data(filename, train_data) -> Tuple[set, dict]:
        samples = Fitipy(filename).read().set()
        hash_to_ind = {
            calc_sample_hash(inp, outp): ind
            for ind, (inp, outp) in enumerate(zip(*train_data))
        }
        for hsh in list(samples):
            if hsh not in hash_to_ind:
                print('Missing hash:', hsh)
                samples.remove(hsh)
        return samples, hash_to_ind

    @staticmethod
    def load_data(args: Any) -> Tuple[tuple, tuple]:
        data = TrainData.from_both(args.tags_file, args.tags_folder, args.folder)
        print('Data:', data)
        train, test = data.load(True, not args.no_validation)

        print('Inputs shape:', train[0].shape)
        print('Outputs shape:', train[1].shape)

        if test:
            print('Test inputs shape:', test[0].shape)
            print('Test outputs shape:', test[1].shape)

        if 0 in train[0].shape or 0 in train[1].shape:
            print('Not enough data to train')
            exit(1)

        return train, test

    @property
    def sampled_data(self):
        """Returns (train_inputs, train_outputs)"""
        if self.args.samples_file:
            if self.args.invert_samples:
                chosen_samples = set(self.hash_to_ind) - self.samples
            else:
                chosen_samples = self.samples
            selected_indices = [self.hash_to_ind[h] for h in chosen_samples]
            return self.train[0][selected_indices], self.train[1][selected_indices]
        else:
            return self.train[0], self.train[1]

    def run(self):
        self.model.summary()
        train_inputs, train_outputs = self.sampled_data
        self.model.fit(
            train_inputs, train_outputs, self.args.batch_size,
            self.epoch + self.args.epochs, validation_data=self.test,
            initial_epoch=self.epoch, callbacks=self.callbacks
        )


main = TrainScript.run_main

if __name__ == '__main__':
    main()
