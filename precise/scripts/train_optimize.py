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
Use black box optimization to tune model hyperparameters

:-t --trials-name str -
    Filename to save hyperparameter optimization trials in
    '.bbopt.json' will automatically be appended

:-c --cycles int 20
    Number of cycles of optimization to run

:-m --model str .cache/optimized.net
    Model to load from

...
"""
import numpy
# Optimizer blackhat
from glob import glob
from os import remove
from os.path import isfile, splitext, join
from pprint import pprint
from prettyparse import Usage
from shutil import rmtree
from typing import Any

from precise.model import ModelParams, create_model
from precise.scripts.train import TrainScript
from precise.train_data import TrainData


class TrainOptimizeScript(TrainScript):
    Usage(__doc__) | TrainScript.usage

    def __init__(self, args):
        super().__init__(args)
        from bbopt import BlackBoxOptimizer
        self.bb = BlackBoxOptimizer(file=self.args.trials_name)
        if not self.test:
            data = TrainData.from_both(self.args.tags_file, self.args.tags_folder, self.args.folder)
            _, self.test = data.load(False, True)

        from keras.callbacks import ModelCheckpoint
        for i in list(self.callbacks):
            if isinstance(i, ModelCheckpoint):
                self.callbacks.remove(i)

    def process_args(self, args: Any):
        model_parts = glob(splitext(args.model)[0] + '.*')
        if len(model_parts) < 5:
            for name in model_parts:
                if isfile(name):
                    remove(name)
                else:
                    rmtree(name)
        args.trials_name = args.trials_name.replace('.bbopt.json', '').replace('.json', '')
        if not args.trials_name:
            if isfile(join('.cache', 'trials.bbopt.json')):
                remove(join('.cache', 'trials.bbopt.json'))
            args.trials_name = join('.cache', 'trials')

    def run(self):
        print('Writing to:', self.args.trials_name + '.bbopt.json')
        for i in range(self.args.cycles):
            self.bb.run(backend="random")
            print("\n= %d = (example #%d)" % (i + 1, len(self.bb.get_data()["examples"]) + 1))

            params = ModelParams(
                recurrent_units=self.bb.randint("units", 1, 70, guess=50),
                dropout=self.bb.uniform("dropout", 0.1, 0.9, guess=0.6),
                extra_metrics=self.args.extra_metrics,
                skip_acc=self.args.no_validation,
                loss_bias=1.0 - self.args.sensitivity
            )
            print('Testing with:', params)
            model = create_model(self.args.model, params)
            model.fit(
                *self.sampled_data, batch_size=self.args.batch_size,
                epochs=self.epoch + self.args.epochs,
                validation_data=self.test * (not self.args.no_validation),
                callbacks=self.callbacks, initial_epoch=self.epoch
            )
            resp = model.evaluate(*self.test, batch_size=self.args.batch_size)
            if not isinstance(resp, (list, tuple)):
                resp = [resp, None]
            test_loss, test_acc = resp
            predictions = model.predict(self.test[0], batch_size=self.args.batch_size)

            num_false_positive = numpy.sum(predictions * (1 - self.test[1]) > 0.5)
            num_false_negative = numpy.sum((1 - predictions) * self.test[1] > 0.5)
            false_positives = num_false_positive / numpy.sum(self.test[1] < 0.5)
            false_negatives = num_false_negative / numpy.sum(self.test[1] > 0.5)

            from math import exp
            param_score = 1.0 / (1.0 + exp((model.count_params() - 11000) / 2000))
            fitness = param_score * (1.0 - 0.2 * false_negatives - 0.8 * false_positives)

            self.bb.remember({
                "test loss": test_loss,
                "test accuracy": test_acc,
                "false positive%": false_positives,
                "false negative%": false_negatives,
                "fitness": fitness
            })

            print("False positive: ", false_positives * 100, "%")

            self.bb.maximize(fitness)
            pprint(self.bb.get_current_run())
        best_example = self.bb.get_optimal_run()
        print("\n= BEST = (example #%d)" % self.bb.get_data()["examples"].index(best_example))
        pprint(best_example)


main = TrainOptimizeScript.run_main

if __name__ == '__main__':
    main()
