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
Conversion of audio data to predictions using Pocketsphinx
Used for comparison with Precise
"""
import numpy as np
from typing import *
from typing import BinaryIO

from precise.params import pr
from precise.util import audio_to_buffer


class PocketsphinxListener:
    """Pocketsphinx listener implementation used for comparison with Precise"""

    def __init__(self, key_phrase, dict_file, hmm_folder, threshold=1e-90, chunk_size=-1):
        from pocketsphinx import Decoder
        config = Decoder.default_config()
        config.set_string('-hmm', hmm_folder)
        config.set_string('-dict', dict_file)
        config.set_string('-keyphrase', key_phrase)
        config.set_float('-kws_threshold', float(threshold))
        config.set_float('-samprate', 16000)
        config.set_int('-nfft', 2048)
        config.set_string('-logfn', '/dev/null')
        self.key_phrase = key_phrase
        self.buffer = b'\0' * pr.sample_depth * pr.buffer_samples
        self.pr = pr
        self.read_size = -1 if chunk_size == -1 else pr.sample_depth * chunk_size

        try:
            self.decoder = Decoder(config)
        except RuntimeError:
            options = dict(key_phrase=key_phrase, dict_file=dict_file,
                           hmm_folder=hmm_folder, threshold=threshold)
            raise RuntimeError('Invalid Pocketsphinx options: ' + str(options))

    def _transcribe(self, byte_data):
        self.decoder.start_utt()
        self.decoder.process_raw(byte_data, False, False)
        self.decoder.end_utt()
        return self.decoder.hyp()

    def found_wake_word(self, frame_data):
        hyp = self._transcribe(frame_data + b'\0' * int(2 * 16000 * 0.01))
        return bool(hyp and self.key_phrase in hyp.hypstr.lower())

    def update(self, stream: Union[BinaryIO, np.ndarray, bytes]) -> float:
        if isinstance(stream, np.ndarray):
            chunk = audio_to_buffer(stream)
        else:
            if isinstance(stream, (bytes, bytearray)):
                chunk = stream
            else:
                chunk = stream.read(self.read_size)
            if len(chunk) == 0:
                raise EOFError
        self.buffer = self.buffer[len(chunk):] + chunk
        return float(self.found_wake_word(self.buffer))
