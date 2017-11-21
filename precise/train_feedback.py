#!/usr/bin/env python3
# This script trains the network, selectively choosing
# segments from data/random that cause an activation. These
# segments are moved into data/not-keyword and the network is retrained

import sys
sys.path += ['.']  # noqa

import argparse
from glob import glob
from os.path import basename, splitext

from precise.common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs to train before continueing evaluation')
    parser.add_argument('-j', '--jump', type=int, default=2, help='Number of features to skip while evaluating')
    parser.add_argument('-s', '--skip-trained', type=bool, default=True, help='Whether to skip random files that have already been trained on')
    parser.add_argument('-l', '--load', type=bool, default=True)
    parser.add_argument('-b', '--save-best', type=bool, default=False)
    parser.add_argument('-o', '--out-dir', default='data/not-keyword/generated')
    parser.add_argument('-d', '--data-dir', default='data')
    parser.add_argument('-r', '--random-data-dir', default='data/random')
    parser.add_argument('-m', '--model', default='keyword.net')

    args = parser.parse_args()

    trained_fns = []
    if isfile('keyword.trained.txt'):
        with open('keyword.trained.txt', 'r') as f:
            trained_fns = f.read().split('\n')

    random_inputs = ((f, load_audio(f)) for f in glob(args.random_data_dir + '/*.wav') if not args.skip_trained or f not in trained_fns)
    validation_data = load_data(args.data_dir + '/test')

    model = create_model(args.model, args.load)

    from keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(args.model, monitor='val_acc', save_best_only=args.save_best, mode='max')

    makedirs(args.out_dir, exist_ok=True)

    try:
        for fn, full_audio in random_inputs:
            features = vectorize_raw(full_audio[:pr.buffer_samples])
            counter = 0

            print('Starting file ' + fn + '...')
            for i in range(pr.buffer_samples - pr.buffer_samples % pr.hop_samples, len(full_audio), pr.hop_samples):
                print('\r' + str(i * 100. / len(full_audio)) + '%', end='', flush=True)
                window = full_audio[i-pr.window_samples:i]

                vec = vectorize_raw(window)
                assert len(vec) == 1
                features = np.concatenate([features[1:], vec])

                counter += 1
                if counter % args.jump == 0:
                    out = model.predict(np.array([features]))[0]
                else:
                    out = 0.0
                if out > 0.5:
                    folder = args.out_dir
                    if args.epochs > 0 and (counter // args.jump) % 5 == 0:
                        folder = join(args.data_dir, 'test', 'not-keyword', 'generated')
                    name = join(folder, splitext(basename(fn))[0] + '-' + str(i)) + '.wav'
                    audio = full_audio[i - pr.buffer_samples:i]
                    audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
                    wavio.write(name, audio, pr.sample_rate, sampwidth=pr.sample_depth, scale='none')
                    print('\nSaving to:', name)

                    if args.epochs > 0:
                        inputs, outputs = load_data(args.data_dir)
                        model.fit(inputs, outputs, 5000, 1, validation_data=validation_data)
                        model.save('keyword.net')
                        model.fit(inputs, outputs, 5000, args.epochs - 1, validation_data=validation_data, callbacks=[checkpoint])
            print()

            with open('keyword.trained.txt', 'a') as f:
                f.write('\n' + fn)

    except KeyboardInterrupt:
        print()

if __name__ == '__main__':
    main()
