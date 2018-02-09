# Python 2 + 3
# Copyright (c) 2017 Mycroft AI Inc.

import atexit
from subprocess import PIPE, Popen
from threading import Thread


class PreciseEngine:
    def __init__(self, exe_file, model_file, chunk_size=2048):
        self.exe_file = exe_file
        self.model_file = model_file
        self.chunk_size = chunk_size
        self.proc = None

    def start(self):
        self.proc = Popen([self.exe_file, self.model_file, str(self.chunk_size)], stdin=PIPE,
                          stdout=PIPE)

    def stop(self):
        if self.proc:
            self.proc.kill()
            self.proc = None

    def get_prediction(self, chunk):
        self.proc.stdin.write(chunk)
        self.proc.stdin.flush()
        return float(self.proc.stdout.readline())


class ListenerEngine:
    def __init__(self, listener):
        self.start = lambda: None
        self.stop = lambda: None
        self.get_prediction = listener.update


class PreciseRunner:
    """
    Wrapper to use Precise

    Args:
        exe_file (str): Location to precise-stream executable
        model (str): Location to .pb model file to use (with .pb.params)
        chunk_size (int): Number of samples per prediction. Higher numbers
                          decrease CPU usage but increase latency
        stream (BinaryIO): Binary audio stream to read 16000 Hz 1 channel int16
                           audio from. If not given, the microphone is used
        on_prediction: callback for every new prediction
        on_activation: callback for when the wake word is heard
    """
    def __init__(self, engine, chunk_size=1024, stream=None,
                 on_prediction=lambda x: None, on_activation=lambda: None, trigger_level=3):
        self.engine = engine
        self.pa = None
        self.chunk_size = chunk_size
        self.thread = None
        self.stream = stream

        self.on_prediction = on_prediction
        self.on_activation = on_activation
        self.running = False
        self.trigger_level = trigger_level
        atexit.register(self.stop)

    def start(self):
        """Start listening from stream"""
        if self.stream is None:
            from pyaudio import PyAudio, paInt16
            self.pa = PyAudio()
            self.stream = self.pa.open(16000, 1, paInt16, True, frames_per_buffer=self.chunk_size)

        self.engine.start()
        self.running = True
        self.thread = Thread(target=self._handle_predictions)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop listening and close stream"""
        self.engine.stop()

        if self.thread:
            self.running = False
            self.thread.join()
            self.thread = None

        if self.pa:
            self.pa.terminate()
            self.stream.stop_stream()
            self.stream = self.pa = None

    def _handle_predictions(self):
        """Continuously check Precise process output"""
        activation = 0
        while self.running:
            chunk = self.stream.read(self.chunk_size)
            prob = self.engine.get_prediction(chunk)
            self.on_prediction(prob)

            if prob > 0.5 or activation < 0:
                activation += 1
                if activation > self.trigger_level:
                    activation = -self.chunk_size // 50
                    self.on_activation()
            elif activation > 0:
                activation -= 1
