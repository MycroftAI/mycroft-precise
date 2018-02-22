from typing import *
from typing import BinaryIO
import numpy as np

from precise.util import audio_to_buffer
from precise.params import pr


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
        self.decoder = Decoder(config)
        self.buffer = b'\0' * pr.sample_depth * pr.buffer_samples
        self.pr = pr
        self.read_size = -1 if chunk_size == -1 else pr.sample_depth * chunk_size

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
