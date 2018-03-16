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
from setuptools import setup

from precise import __version__

setup(
    name='mycroft-precise',
    version=__version__,
    packages=[
        'precise',
        'precise.scripts',
        'precise.pocketsphinx',
        'precise.pocketsphinx.scripts'
    ],
    entry_points={
        'console_scripts': [
            'precise-collect=precise.scripts.collect:main',
            'precise-convert=precise.scripts.convert:main',
            'precise-eval=precise.scripts.eval:main',
            'precise-record=precise.scripts.record:main',
            'precise-record-pocketsphinx=precise.pocketsphinx.scripts.record:main',
            'precise-engine=precise.scripts.engine:main',
            'precise-test=precise.scripts.test:main',
            'precise-test-pocketsphinx=precise.pocketsphinx.scripts.test:main',
            'precise-train=precise.scripts.train:main',
            'precise-train-incremental=precise.scripts.train_incremental:main',
        ]
    },
    install_requires=[
        'numpy',
        'tensorflow',
        'speechpy',
        'pyaudio',
        'keras',
        'h5py',
        'wavio',
        'typing',
        'prettyparse',
        'precise-runner'
    ],

    author='Matthew Scholefield',
    author_email='matthew.scholefield@mycroft.ai',
    description='Mycroft Precise Wake Word Listener',
    keywords='wakeword keyword wake word listener sound',
    url='http://github.com/MycroftAI/mycroft-precise',

    zip_safe=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
