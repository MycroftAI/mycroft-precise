#!/usr/bin/env python3
# Attribution: This script was adapted from https://github.com/amir-abdi/keras_to_tensorflow
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
Convert wake word model from Keras to TensorFlow

:model str
    Input Keras model (.net)

:-o --out str {model}.pb
    Custom output TensorFlow protobuf filename
"""
import os
from os.path import split, isfile
from prettyparse import Usage
from shutil import copyfile

from precise.scripts.base_script import BaseScript


class ConvertScript(BaseScript):
    usage = Usage(__doc__)

    def run(self):
        args = self.args
        model_name = args.model.replace('.net', '')
        self.convert(args.model, args.out.format(model=model_name))

    def convert(self, model_path: str, out_file: str):
        """
        Converts an HD5F file from Keras to a .pb for use with TensorFlow

        Args:
            model_path: location of Keras model
            out_file: location to write protobuf
        """
        print('Converting', model_path, 'to', out_file, '...')

        import tensorflow as tf
        from precise.model import load_precise_model
        from keras import backend as K

        out_dir, filename = split(out_file)
        out_dir = out_dir or '.'
        os.makedirs(out_dir, exist_ok=True)

        K.set_learning_phase(0)
        model = load_precise_model(model_path)

        out_name = 'net_output'
        tf.identity(model.output, name=out_name)
        print('Output node name:', out_name)
        print('Output folder:', out_dir)

        sess = K.get_session()

        # Write the graph in human readable
        tf.train.write_graph(sess.graph.as_graph_def(), out_dir, filename + 'txt', as_text=True)
        print('Saved readable graph to:', filename + 'txt')

        # Write the graph in binary .pb file
        from tensorflow.python.framework import graph_util
        from tensorflow.python.framework import graph_io

        cgraph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [out_name])
        graph_io.write_graph(cgraph, out_dir, filename, as_text=False)

        if isfile(model_path + '.params'):
            copyfile(model_path + '.params', out_file + '.params')

        print('Saved graph to:', filename)

        del sess


main = ConvertScript.run_main

if __name__ == '__main__':
    main()
