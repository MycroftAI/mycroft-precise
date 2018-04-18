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
import json
from argparse import ArgumentParser
from contextlib import suppress
from glob import glob
from hashlib import md5
from os.path import join, isfile, dirname
from typing import *

import numpy as np
from prettyparse import add_to_parser

from precise.util import find_wavs
from precise.vectorization import load_vector, vectorize_inhibit, vectorize


class TrainData:
    """Class to handle loading of wave data from categorized folders and tagged text files"""

    def __init__(self, train_files: Tuple[List[str], List[str]],
                 test_files: Tuple[List[str], List[str]]):
        self.train_files, self.test_files = train_files, test_files

    @classmethod
    def from_folder(cls, folder: str) -> 'TrainData':
        """
        Load a set of data from a structured folder in the following format:
        {prefix}/
            wake-word/
                *.wav
            not-wake-word/
                *.wav
            test/
                wake-word/
                    *.wav
                not-wake-word/
                    *.wav
        """
        return cls(find_wavs(folder), find_wavs(join(folder, 'test')))

    @classmethod
    def from_tags(cls, tags_file: str, tags_folder: str) -> 'TrainData':
        """
        Load a set of data from a text file with tags in the following format:
            <file_id>  (tab)  <tag>
            <file_id>  (tab)  <tag>

            file_id: identifier of file such that the following
                     file exists: {tags_folder}/{data_id}.wav
            tag: "wake-word" or "not-wake-word"
        """
        if not tags_file:
            num_ignored_wavs = len(glob(join(tags_folder, '*.wav')))
            if num_ignored_wavs > 10:
                print('WARNING: Found {} wavs but no tags file specified!'.format(num_ignored_wavs))
            return cls(([], []), ([], []))
        if not isfile(tags_file):
            raise RuntimeError('Database file does not exist: ' + tags_file)

        train_groups = {}
        train_group_file = join(tags_file.replace('.txt', '') + '.groups.json')
        if isfile(train_group_file):
            with open(train_group_file) as f:
                train_groups = json.load(f)

        tags_files = {
            'wake-word': [],
            'not-wake-word': []
        }
        with open(tags_file) as f:
            for line in f.read().split('\n'):
                if not line:
                    continue
                file, tag = line.split('\t')
                tags_files[tag.strip()].append(join(tags_folder, file.strip() + '.wav'))

        train_files, test_files = ([], []), ([], [])
        for label, rows in enumerate([tags_files['wake-word'], tags_files['not-wake-word']]):
            for fn in rows:
                if not isfile(fn):
                    print('Missing file:', fn)
                    continue
                if fn not in train_groups:
                    train_groups[fn] = (
                        'test' if md5(fn.encode('utf8')).hexdigest() > 'c' * 32
                        else 'train'
                    )
                {
                    'train': train_files,
                    'test': test_files
                }[train_groups[fn]][label].append(fn)

        with open(train_group_file, 'w') as f:
            json.dump(train_groups, f)

        return cls(train_files, test_files)

    @classmethod
    def from_both(cls, tags_file: str, tags_folder: str, folder: str) -> 'TrainData':
        """Load data from both a database and a structured folder"""
        return cls.from_tags(tags_file, tags_folder) + cls.from_folder(folder)

    def load(self, train=True, test=True) -> tuple:
        """
        Load the vectorized representations of the stored data files
        Args:
            train: Whether to load train data
            test: Whether to load test data
        """
        return self.__load(self.__load_files, train, test)

    def load_inhibit(self, train=True, test=True) -> tuple:
        """Generate data with inhibitory inputs created from wake word samples"""

        def loader(kws: list, nkws: list):
            from precise.params import pr
            inputs = np.empty((0, pr.n_features, pr.feature_size))
            outputs = np.zeros((len(kws), 1))
            for f in kws:
                if not isfile(f):
                    continue
                new_vec = load_vector(f, vectorize_inhibit)
                inputs = np.concatenate([inputs, new_vec])

            return self.merge((inputs, outputs), self.__load_files(kws, nkws))

        return self.__load(loader, train, test)

    @staticmethod
    def merge(data_a: tuple, data_b: tuple) -> tuple:
        return np.concatenate((data_a[0], data_b[0])), np.concatenate((data_a[1], data_b[1]))

    @staticmethod
    def parse_args(parser: ArgumentParser) -> Any:
        """Return parsed args from parser, adding options for train data inputs"""
        extra_usage = '''
            :folder str
                Folder to wav files from
            
            :-tf --tags-folder str {folder}
                Specify a different folder to load file ids
                in tags file from
            
            :-tg --tags-file str -
                Text file to load tags from where each line is
                <file_id> TAB (wake-word|not-wake-word) and
                {folder}/<file_id>.wav exists
            
        '''
        add_to_parser(parser, extra_usage)
        args = parser.parse_args()
        args.tags_folder = args.tags_folder.format(folder=args.folder)
        return args

    def __repr__(self) -> str:
        string = '<TrainData wake_words={kws} not_wake_words={nkws}' \
                 ' test_wake_words={test_kws} test_not_wake_words={test_nkws}>'
        return string.format(
            kws=len(self.train_files[0]), nkws=len(self.train_files[1]),
            test_kws=len(self.test_files[0]), test_nkws=len(self.test_files[1])
        )

    def __add__(self, other: 'TrainData') -> 'TrainData':
        if not isinstance(other, TrainData):
            raise TypeError('Can only add TrainData to TrainData')
        return TrainData((self.train_files[0] + other.train_files[0],
                          self.train_files[1] + other.train_files[1]),
                         (self.test_files[0] + other.test_files[0],
                          self.test_files[1] + other.test_files[1]))

    def __load(self, loader: Callable, train: bool, test: bool) -> tuple:
        return tuple([
            loader(*files) if files else None
            for files in (train and self.train_files,
                          test and self.test_files)
        ])

    @staticmethod
    def __load_files(kw_files: list, nkw_files: list, vectorizer: Callable = vectorize) -> tuple:
        inputs = []
        outputs = []

        def add(filenames, output):
            for f in filenames:
                try:
                    inputs.append(load_vector(f, vectorizer))
                    outputs.append(np.array([output]))
                except ValueError:
                    print('Skipping invalid file:', f)

        print('Loading wake-word...')
        add(kw_files, 1.0)

        print('Loading not-wake-word...')
        add(nkw_files, 0.0)

        from precise.params import pr
        return (
            np.array(inputs) if inputs else np.empty((0, pr.n_features, pr.feature_size)),
            np.array(outputs) if outputs else np.empty((0, 1))
        )
