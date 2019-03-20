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
from argparse import ArgumentParser
from precise.util import activate_notify
from precise_runner import PreciseRunner, PreciseEngine
from threading import Event


def main():
    parser = ArgumentParser('Implementation demo of precise-engine')
    parser.add_argument('engine', help='Location of binary engine file')
    parser.add_argument('model')
    args = parser.parse_args()

    def on_prediction(prob):
        print('!' if prob > 0.5 else '.', end='', flush=True)

    def on_activation():
        activate_notify()

    engine = PreciseEngine(args.engine, args.model)
    PreciseRunner(engine, on_prediction=on_prediction, on_activation=on_activation,
                  trigger_level=0).start()
    Event().wait()  # Wait forever


if __name__ == '__main__':
    main()
