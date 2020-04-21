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
Train a model, sampling data points with the highest loss from a larger dataset

:-c --cycles int 200
    Number of sampling cycles of size {epoch} to run

:-n --num-sample-chunk int 50
    Number of new samples to introduce at a time between training cycles

:-sf --samples-file str -
    Json file to write selected samples to.
    Default = {model_base}.samples.json

:-is --invert-samples
    Unused parameter
...
"""
from itertools import islice

from fitipy import Fitipy
from prettyparse import Usage

from precise.scripts.train import TrainScript
from precise.util import calc_sample_hash


class TrainSampledScript(TrainScript):
    usage = Usage(__doc__) | TrainScript.usage

    def __init__(self, args):
        super().__init__(args)
        if self.args.invert_samples:
            raise ValueError('--invert-samples should be left blank')
        self.args.samples_file = (self.args.samples_file or '{model_base}.samples.json').format(
            model_base=self.model_base
        )
        self.samples, self.hash_to_ind = self.load_sample_data(self.args.samples_file, self.train)
        self.metrics_fiti = Fitipy(self.model_base + '.logs', 'sampling-metrics.txt')

    def write_sampling_metrics(self, predicted):
        correct = float(sum((predicted > 0.5) == (self.train[1] > 0.5)) / len(self.train[1]))
        print('Successfully calculated: {0:.3%}'.format(correct))

        lines = self.metrics_fiti.read().lines()
        lines.append('{}\t{}'.format(len(self.samples) / len(self.train[1]), correct))
        self.metrics_fiti.write().lines(lines)

    def choose_new_samples(self, predicted):
        failed_samples = {
            calc_sample_hash(inp, target)
            for i, (inp, pred, target) in enumerate(zip(self.train[0], predicted, self.train[1]))
            if (pred > 0.5) != (target > 0.5)
        }
        remaining_failed_samples = failed_samples - self.samples
        print('Remaining failed samples:', len(remaining_failed_samples))
        return islice(remaining_failed_samples, self.args.num_sample_chunk)

    def run(self):
        print('Writing to:', self.args.samples_file)
        print('Writing metrics to:', self.metrics_fiti.path)
        for _ in range(self.args.cycles):
            print('Calculating on whole dataset...')
            predicted = self.model.predict(self.train[0])

            self.samples.update(self.choose_new_samples(predicted))
            Fitipy(self.args.samples_file).write().set(self.samples)
            print('Added', self.args.num_sample_chunk, 'samples')

            self.write_sampling_metrics(predicted)

            self.model.fit(
                *self.sampled_data, batch_size=self.args.batch_size,
                epochs=self.epoch + self.args.epochs,
                callbacks=self.callbacks, initial_epoch=self.epoch,
                validation_data=self.test
            )


main = TrainSampledScript.run_main

if __name__ == '__main__':
    main()
