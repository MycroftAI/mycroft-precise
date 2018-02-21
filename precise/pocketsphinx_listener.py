

class PocketsphinxListener:
    def __init__(self, key_phrase, dict_file, hmm_folder, threshold=1e-90):
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

    def transcribe(self, byte_data):
        self.decoder.start_utt()
        self.decoder.process_raw(byte_data, False, False)
        self.decoder.end_utt()
        return self.decoder.hyp()

    def found_wake_word(self, frame_data):
        hyp = self.transcribe(frame_data + b'\0' * int(2 * 16000 * 0.01))
        return bool(hyp and self.key_phrase in hyp.hypstr.lower())
