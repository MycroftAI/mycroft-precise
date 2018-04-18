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
import wave
from subprocess import check_output, PIPE

from prettyparse import create_parser

from precise.pocketsphinx.listener import PocketsphinxListener
from precise.scripts.test import show_stats, Stats
from precise.train_data import TrainData

usage = '''
    Test a dataset using Pocketsphinx
    
    :key_phrase str
        Key phrase composed of words from dictionary
    
    :dict_file str
        Filename of dictionary with word pronunciations
    
    :hmm_folder str
        Folder containing hidden markov model
    
    :-th --threshold str 1e-90
        Threshold for activations
    
    :-t --use-train
        Evaluate training data instead of test data
    
    :-nf --no-filenames
        Don't show the names of files that failed
    
    ...
'''


def eval_file(filename) -> float:
    transcription = check_output(
        ['pocketsphinx_continuous', '-kws_threshold', '1e-20', '-keyphrase', 'hey my craft',
         '-infile', filename], stderr=PIPE)
    return float(bool(transcription) and not transcription.isspace())


def test_pocketsphinx(listener: PocketsphinxListener, data_files) -> Stats:
    def run_test(filenames, name):
        print()
        print('===', name, '===')
        negatives, positives = [], []
        for filename in filenames:
            try:
                with wave.open(filename) as wf:
                    frames = wf.readframes(wf.getnframes())
            except (OSError, EOFError):
                print('?', end='', flush=True)
                continue
            out = listener.found_wake_word(frames)
            {False: negatives, True: positives}[out].append(filename)
            print('!' if out else '.', end='', flush=True)
        print()
        return negatives, positives

    false_neg, true_pos = run_test(data_files[0], 'Wake Word')
    true_neg, false_pos = run_test(data_files[1], 'Not Wake Word')
    return Stats(false_pos, false_neg, true_pos, true_neg)


def main():
    args = TrainData.parse_args(create_parser(usage))
    data = TrainData.from_both(args.tags_file, args.tags_folder, args.folder)
    data_files = data.train_files if args.use_train else data.test_files
    listener = PocketsphinxListener(
        args.key_phrase, args.dict_file, args.hmm_folder, args.threshold
    )

    print('Data:', data)
    stats = test_pocketsphinx(listener, data_files)
    show_stats(stats, not args.no_filenames)


if __name__ == '__main__':
    main()
