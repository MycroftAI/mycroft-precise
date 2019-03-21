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
from functools import partial
from os.path import basename, splitext

from prettyparse import create_parser
from typing import Callable, Tuple

from precise.network_runner import Listener
from precise.params import inject_params, pr
from precise.stats import Stats
from precise.train_data import TrainData
from precise.vectorization import get_cache_folder

usage = '''
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
    
    ...
'''


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
        if get_cache_folder() != self.prev_cache:
            self.prev_cache = get_cache_folder()
            self.data = self.loader()
        return self.data


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Please install matplotlib first')
        raise SystemExit(2)

    parser = create_parser(usage)
    parser.add_argument('models', nargs='+', help='Either Keras (.net) or TensorFlow (.pb) models to test')
    args = TrainData.parse_args(parser)

    data = TrainData.from_both(args.tags_file, args.tags_folder, args.folder)
    filenames = sum(data.train_files if args.use_train else data.test_files, [])
    loader = CachedDataLoader(partial(
        data.load, args.use_train, not args.use_train, shuffle=False
    ))

    for model in args.models:
        train, test = loader.load_for(model)
        inputs, targets = train if args.use_train else test
        predictions = Listener.find_runner(model)(model).predict(inputs)

        print('Generating statistics...')
        stats = Stats(predictions, targets, filenames)
        print('\n' + stats.counts_str() + '\n\n' + stats.summary_str() + '\n')

        thresholds = get_thresholds(args.resolution, args.power)
        print('Generating x values...')
        x = [stats.false_positives(i) for i in thresholds]
        print('Generating y values...')
        y = [stats.false_negatives(i) for i in thresholds]

        plt.plot(x, y, marker='x', linestyle='-', label=basename(splitext(model)[0]))

        if args.labels:
            for x, y, threshold in zip(x, y, thresholds):
                plt.annotate('{:.4f}'.format(threshold), (x, y))

    print('Data:', data)
    plt.legend()
    plt.xlabel('False Positives')
    plt.ylabel('False Negatives')
    plt.show()


if __name__ == '__main__':
    main()
