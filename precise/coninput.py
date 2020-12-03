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
Platform independent console Input abstraction
"""
from abc import abstractmethod

from os import name as os_name

from typing import AnyStr

if os_name == 'nt':
    import msvcrt
else:
    from select import select
    from sys import stdin
    from termios import tcsetattr, tcgetattr, TCSADRAIN
    import tty


class InputInterface:

    @abstractmethod
    def key_pressed(self) -> bool:
        pass

    @abstractmethod
    def show_input(self):
        pass

    @abstractmethod
    def hide_input(self):
        pass

    @abstractmethod
    def read_key(self) -> AnyStr:
        pass


class POSIX(InputInterface):

    def __init__(self):
        self.orig_settings = tcgetattr(stdin)

    def key_pressed(self) -> bool:
        return select([stdin], [], [], 0) == ([stdin], [], [])

    def show_input(self):
        tcsetattr(stdin, TCSADRAIN, self.orig_settings)

    def hide_input(self):
        tty.setcbreak(stdin.fileno())

    def read_key(self) -> AnyStr:
        return stdin.read(1)


class Windows(InputInterface):

    def __init__(self):
        self.input_shown = True

    def key_pressed(self) -> bool:
        return msvcrt.kbhit()

    def show_input(self):
        self.input_shown = True
        pass

    def hide_input(self):
        self.input_shown = False
        pass

    def read_key(self) -> AnyStr:
        if self.input_shown:
            char = msvcrt.getche()
        else:
            char = msvcrt.getch()
        return char.decode('utf-8')


def get_input() -> InputInterface:
    if os_name == "nt":
        return Windows()
    else:
        return POSIX()
