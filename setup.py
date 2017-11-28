#!/usr/bin/env python3

from setuptools import setup, find_packages
from precise import __version__

setup(
    name='mycroft-precise',
    version=__version__,
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'precise-train=precise.train:main',
            'precise-train-feedback=precise.train_feedback:main',
            'precise-stream=precise.stream:main',
            'precise-test=precise.test:main',
            'precise-convert=precise.convert:main'
        ]
    },
    install_requires=[
        'numpy',
        'tensorflow',
        'speechpy',
        'pyaudio',
        'keras',
        'wavio',
        # 'precise-runner'  # Needs to get uploaded to PyPi first
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
