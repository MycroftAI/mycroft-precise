#!/usr/bin/env python3
# Copyright (c) 2017 Mycroft AI Inc.
import wave
from subprocess import check_output, PIPE
from prettyparse import create_parser

from precise.pocketsphinx.listener import PocketsphinxListener
from precise.scripts.test import show_stats
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
    transcription = check_output(['pocketsphinx_continuous', '-kws_threshold', '1e-20', '-keyphrase', 'hey my craft', '-infile', filename], stderr=PIPE)
    return float(bool(transcription) and not transcription.isspace())


def main():
    args = TrainData.parse_args(create_parser(usage))
    data = TrainData.from_both(args.db_file, args.db_folder, args.data_dir)
    print('Data:', data)

    listener = PocketsphinxListener(args.key_phrase, args.dict_file, args.hmm_folder, args.threshold)

    def run_test(filenames, name):
        print()
        print('===', name, '===')
        negatives, positives = [], []
        for filename in filenames:
            with wave.open(filename) as wf:
                frames = wf.readframes(wf.getnframes())
            out = listener.found_wake_word(frames)
            {False: negatives, True: positives}[out].append(filename)
            print('!' if out else '.', end='', flush=True)
        print()
        return negatives, positives

    data_files = data.train_files if args.use_train else data.test_files
    false_neg, true_pos = run_test(data_files[0], 'Keyword')
    true_neg, false_pos = run_test(data_files[1], 'Not Keyword')

    show_stats(false_pos, false_neg, true_pos, true_neg, not args.no_filenames)


if __name__ == '__main__':
    main()
