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
import atexit

from fitipy import Fitipy
from time import sleep

import wavio
from os.path import join, basename, splitext

from glob import glob
from prettyparse import create_parser
from pyaudio import PyAudio, paFloat32
from random import shuffle
from subprocess import Popen
from tkinter import *
from tkinter.ttk import *
from threading import Thread, Event


from precise.util import load_audio
from precise.params import pr

usage = '''
    Tag samples with a variable score
    :folder str
        Folder to load wavs from

    :tags_file str
        File to write tags to
'''
def play_wav(filename, p: PyAudio):
    audio = load_audio(filename)



def main():
    args = create_parser(usage).parse_args()
    filenames = glob(join(args.folder, '*.wav'))
    shuffle(filenames)
    wav_id = -1
    stream = None
    stop_event = Event()
    stop_event.set()
    p = PyAudio()
    atexit.register(p.terminate)

    def play_audio(audio_file):
        nonlocal stream
        if stream:
            stop_event.clear()
            stop_event.wait()
            stream.stop_stream()
            stream.close()
            stream = None
        audio = load_audio(audio_file)[-pr.buffer_samples:]
        audio /= 2 * min(audio.mean() + 4 * audio.std(), abs(audio).max())
        stream = p.open(format=paFloat32, channels=1, rate=pr.sample_rate, output=True)
        stream.start_stream()
        def write_audio():
            data = audio.astype('float32').tostring()
            chunk_size = 1024
            for pos in range(chunk_size, len(data) + chunk_size, chunk_size):
                if not stop_event.is_set():
                    stop_event.set()
                    return
                stream.write(data[pos - chunk_size:pos])
            while stop_event.is_set():
                sleep(chunk_size / pr.sample_rate)
            stop_event.set()
        Thread(target=write_audio, daemon=True).start()

    tags_file = Fitipy(args.tags_file)
    tags = tags_file.read().dict()
    def submit():
        nonlocal wav_id
        if wav_id >= 0:
            tags[basename(splitext(filenames[wav_id])[0])] = float(slider.get())
            tags_file.write().dict(tags)
        wav_id += 1
        play_audio(filenames[wav_id])

    submit()

    master = Tk()
    label = Label(master, text='0')
    label.pack()
    def on_slider_change(x):
        label['text'] = str(int(float(x)))
    slider = Scale(master, from_=0, to=100, command=on_slider_change)
    slider.pack()

    Button(master, text='Submit', command=submit).pack()
    Button(master, text='Replay', command=lambda: play_audio(filenames[wav_id])).pack()
    mainloop()
    stream.stop_stream()
    stream.close()


if __name__ == '__main__':
    main()
