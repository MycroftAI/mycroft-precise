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
Create a duplicate dataset with added noise

:folder str
    Folder containing source dataset

:-tg --tags-file str -
    Tags file to optionally load from

:noise_folder str
    Folder with wav files containing noise to be added

:output_folder str
    Folder to write the duplicate generated dataset

:-if --inflation-factor int 1
    The number of noisy samples generated per single source sample

:-nl --noise-ratio-low float 0.0
    Minimum random ratio of noise to sample. 1.0 is all noise, no sample sound

:-nh --noise-ratio-high float 0.4
    Maximum random ratio of noise to sample. 1.0 is all noise, no sample sound
"""
from math import sqrt

import numpy as np
import os
from glob import glob
from os import makedirs
from os.path import join, dirname, abspath, splitext
import shutil
from prettyparse import Usage
from random import random

from precise.scripts.base_script import BaseScript
from precise.train_data import TrainData
from precise.util import load_audio
from precise.util import save_audio


class NoiseData:
    def __init__(self, noise_folder: str):
        self.noise_data = [
            load_audio(file)
            for file in glob(join(noise_folder, '*.wav'))
        ]
        self.noise_data_id = 0
        self.noise_pos = 0
        self.repeat_count = 0

    def get_fresh_noise(self, n: int) -> np.ndarray:
        noise_audio = np.empty(0)
        while len(noise_audio) < n:
            noise_source = self.noise_data[self.noise_data_id]
            noise_chunk = noise_source[self.noise_pos:self.noise_pos + n - len(noise_audio)]
            self.noise_pos += n - len(noise_audio)
            if self.noise_pos >= len(noise_source):
                self.noise_pos = 0
                self.noise_data_id += 1
                if self.noise_data_id >= len(self.noise_data):
                    self.noise_data_id = 0
                    self.repeat_count += 1
                    if self.repeat_count == 100:
                        print('Warning: Repeating noise 100+ times. Add more to prevent '
                              'overfitting.')

            noise_audio = np.concatenate([noise_audio, noise_chunk])
        return noise_audio

    def noised_audio(self, audio: np.ndarray, noise_ratio: float) -> np.ndarray:
        noise_data = self.get_fresh_noise(len(audio))
        audio_volume = sqrt(sum(audio ** 2))
        noise_volume = sqrt(sum(noise_data ** 2))
        adjusted_noise = audio_volume * noise_data / noise_volume
        return noise_ratio * adjusted_noise + (1.0 - noise_ratio) * audio


class AddNoiseScript(BaseScript):
    usage = Usage(
        __doc__,
        tags_file=lambda args: abspath(args.tags_file) if args.tags_file else None,
        folder=lambda args: abspath(args.folder),
        output_folder=lambda args: abspath(args.output_folder)
    )

    def run(self):
        args = self.args
        noise_min, noise_max = args.noise_ratio_low, args.noise_ratio_high

        data = TrainData.from_both(args.tags_file, args.folder, args.folder)
        noise_data = NoiseData(args.noise_folder)
        print('Data:', data)

        def translate_filename(source: str, n=0) -> str:
            assert source.startswith(args.folder)
            relative_file = source[len(args.folder):].strip(os.path.sep)
            if n > 0:
                base, ext = splitext(relative_file)
                relative_file = base + '.' + str(n) + ext
            return join(args.output_folder, relative_file)

        all_filenames = sum(data.train_files + data.test_files, [])
        for i, filename in enumerate(all_filenames):
            print('{0:.2%}  \r'.format(i / (len(all_filenames) - 1)), end='', flush=True)

            audio = load_audio(filename)
            for n in range(args.inflation_factor):
                altered = noise_data.noised_audio(audio, noise_min + (noise_max - noise_min) * random())
                output_filename = translate_filename(filename, n)

                makedirs(dirname(output_filename), exist_ok=True)
                save_audio(output_filename, altered)

        print('Done!')

        if args.tags_file and args.tags_file.startswith(args.folder):
            shutil.copy2(args.tags_file, translate_filename(args.tags_file))


main = AddNoiseScript.run_main

if __name__ == '__main__':
    main()
