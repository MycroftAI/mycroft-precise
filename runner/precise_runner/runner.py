# Python 2 + 3
# Copyright (c) 2017 Mycroft AI Inc.

import atexit
from psutil import Popen
from subprocess import PIPE
from threading import Thread


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
    def __init__(self, exe_file, model, chunk_size=1024, stream=None,
                 on_prediction=lambda x: None, on_activation=lambda: None):
        self.pa = None
        self.stream = stream
        self.exe_file = exe_file
        self.model = model
        self.chunk_size = chunk_size
        self.thread = None
        self.proc = None
        self.on_prediction = on_prediction
        self.on_activation = on_activation
        self.running = False
        self.cooldown = 0
        atexit.register(self.stop)

    def start(self):
        """Start listening from stream"""
        if self.stream is None:
            from pyaudio import PyAudio, paInt16
            self.pa = PyAudio()
            self.stream = self.pa.open(16000, 1, paInt16, True, frames_per_buffer=self.chunk_size)

        self.proc = Popen([self.exe_file, self.model, str(self.chunk_size)], stdin=PIPE, stdout=PIPE)
        self.running = True
        self.thread = Thread(target=self._check_output)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop listening and close stream"""
        if self.thread:
            self.running = False
            self.thread.join()
            self.thread = None

        if self.proc:
            self.proc.kill()
            self.proc = None

        if self.pa:
            self.pa.terminate()
            self.stream.stop_stream()
            self.stream = self.pa = None

    def _check_output(self):
        """Continuously check Precise process output"""
        while self.running:
            chunk = self.stream.read(self.chunk_size)
            self.proc.stdin.write(chunk)
            self.proc.stdin.flush()

            prob = float(self.proc.stdout.readline())
            self.on_prediction(prob)

            if self.cooldown > 0:
                self.cooldown -= 1
            elif prob > 0.5:
                self.cooldown = self.chunk_size // 50
                self.on_activation()
