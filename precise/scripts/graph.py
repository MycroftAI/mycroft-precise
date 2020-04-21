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
# limitations under the License
"""
Show ROC curves for a series of models

...

:-t --use-train
    Evaluate training data instead of test data

:-nf --no-filenames
    Don't print out the names of files that failed

:-r --resolution int 100
    Number of points to generate

:-p --power float 3.0
    Power of point distribution

:-l --labels
    Print labels attached to each point

:-o --output-file str -
    File to write data instead of displaying it

:-i --input-file str -
    File to read data from and visualize

...
"""
import numpy as np
from functools import partial
from os.path import basename, splitext
from prettyparse import Usage
from typing import Callable, Tuple

from precise.network_runner import Listener
from precise.params import inject_params, pr
from precise.scripts.base_script import BaseScript
from precise.stats import Stats
from precise.threshold_decoder import ThresholdDecoder
from precise.train_data import TrainData


def get_thresholds(points=100, power=3) -> list:
    """Run a function with a series of thresholds between 0 and 1"""
    return [(i / (points + 1)) ** power for i in range(1, points + 1)]


class CachedDataLoader:
    """
    Class for reloading train data every time the params change

    Args:
        loader: Function that loads the train data (something that calls TrainData.load)
    """

    def __init__(self, loader: Callable):
        self.prev_cache = None
        self.data = None
        self.loader = loader

    def load_for(self, model: str) -> Tuple[list, list]:
        """Injects the model parameters, reloading if they changed, and returning the data"""
        inject_params(model)
        if self.prev_cache != pr.vectorization_md5_hash():
            self.prev_cache = pr.vectorization_md5_hash()
            self.data = self.loader()
        return self.data


def load_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print('Please install matplotlib first')
        raise SystemExit(2)


def calc_stats(model_files, loader, use_train, filenames):
    model_data = {}
    for model in model_files:
        train, test = loader.load_for(model)
        inputs, targets = train if use_train else test
        print('Running network...')
        predictions = Listener.find_runner(model)(model).predict(inputs)
        print(inputs.shape, targets.shape)

        print('Generating statistics...')
        stats = Stats(predictions, targets, filenames)
        print('\n' + stats.counts_str() + '\n\n' + stats.summary_str() + '\n')

        model_name = basename(splitext(model)[0])
        model_data[model_name] = stats
    return model_data


class GraphScript(BaseScript):
    usage = Usage(__doc__)
    usage.add_argument('models', nargs='*', help='Either Keras (.net) or TensorFlow (.pb) models to test')
    usage |= TrainData.usage

    def __init__(self, args):
        super().__init__(args)
        if not args.models and not args.input_file and args.folder:
            args.input_file = args.folder
        if bool(args.models) == bool(args.input_file):
            raise ValueError('Please specify either a list of models or an input file')

        if not args.output_file:
            load_plt()  # Error early if matplotlib not installed

    def run(self):
        args = self.args
        if args.models:
            data = TrainData.from_both(args.tags_file, args.tags_folder, args.folder)
            print('Data:', data)
            filenames = sum(data.train_files if args.use_train else data.test_files, [])
            loader = CachedDataLoader(partial(
                data.load, args.use_train, not args.use_train, shuffle=False
            ))
            model_data = calc_stats(args.models, loader, args.use_train, filenames)
        else:
            model_data = {
                name: Stats.from_np_dict(data) for name, data in np.load(args.input_file)['data'].item().items()
            }
            for name, stats in model_data.items():
                print('=== {} ===\n{}\n\n{}\n'.format(name, stats.counts_str(), stats.summary_str()))

        if args.output_file:
            np.savez(args.output_file, data={name: stats.to_np_dict() for name, stats in model_data.items()})
        else:
            plt = load_plt()
            decoder = ThresholdDecoder(pr.threshold_config, pr.threshold_center)
            thresholds = [decoder.encode(i) for i in np.linspace(0.0, 1.0, args.resolution)[1:-1]]
            for model_name, stats in model_data.items():
                x = [stats.false_positives(i) for i in thresholds]
                y = [stats.false_negatives(i) for i in thresholds]
                plt.plot(x, y, marker='x', linestyle='-', label=model_name)
                if args.labels:
                    for x, y, threshold in zip(x, y, thresholds):
                        plt.annotate('{:.4f}'.format(threshold), (x, y))

            plt.legend()
            plt.xlabel('False Positives')
            plt.ylabel('False Negatives')
            plt.show()


main = GraphScript.run_main

if __name__ == '__main__':
    main()
