from collections import namedtuple

from prettyparse import create_parser

from precise.network_runner import Listener
from precise.params import inject_params
from precise.train_data import TrainData
from select import select
from sys import stdin
import pygame
import time
import os
import shutil
from collections import Counter

usage = '''
    Retag false negatives as wakeword or not wakeword
    
    :model str
        Either Keras (.net) or TensorFlow (.pb) model to test
    
    :-t --use-train
        Evaluate training data instead of test data
    
    :-nf --no-filenames
        Don't print out the names of files that failed
    
    :-ap --append
        Append new tags file to old one
    ...
'''

Stats = namedtuple('Stats', 'false_pos false_neg true_pos true_neg')

def calc_stats(filenames, targets, predictions) -> Stats:
    stats = Stats([], [], [], [])
    for name, target, prediction in zip(filenames, targets, predictions):
        {
            (True, False): stats.false_pos,
            (True, True): stats.true_pos,
            (False, True): stats.false_neg,
            (False, False): stats.true_neg
        }[prediction[0] > 0.5, target[0] > 0.5].append(name)
    return stats

def main():
    args = TrainData.parse_args(create_parser(usage))

    inject_params(args.model)

    data = TrainData.from_both(args.tags_file, args.tags_folder, args.folder)
    train, test = data.load(args.use_train, not args.use_train)
    inputs, targets = train if args.use_train else test

    filenames = sum(data.train_files if args.use_train else data.test_files, [])
    predictions = Listener.find_runner(args.model)(args.model).predict(inputs)
    stats = calc_stats(filenames, targets, predictions)
    false_negatives_array = stats.false_neg
    new_tags = open('tags.txt', 'w')
    

    changed_tags_array = []
    for i in range(0, len(false_negatives_array)):
        pygame.mixer.init(frequency=8000, size=-16, channels=2, buffer=4096)
        pygame.mixer.music.load(false_negatives_array[i])
        pygame.mixer.music.play()
        print('False negative ', (i + 1), ' of ', (len(false_negatives_array)) + 1)
        user_input = input('Enter y if wakeword, enter n for not wakeword \n')
        time.sleep(5)
        false_negatives_array[i] = false_negatives_array[i].lstrip('/Users/madmitrienko/wakewords/files/')
        false_negatives_array[i] = false_negatives_array[i].rstrip('.wav')
        if(user_input == 'y'):
            write_to_tags = '\n' + false_negatives_array[i] + '	wake-word'
            new_tags.write(write_to_tags)

        elif(user_input == 'n'):
            write_to_tags = '\n' + false_negatives_array[i] + '	not-wake-word'          
            new_tags.write(write_to_tags)
    new_tags.close()
    tags.close()
    
if __name__ == '__main__':
    main()
