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
import os
from os.path import split, isfile
from prettyparse import Usage
from shutil import copyfile

from precise.scripts.base_script import BaseScript

class ConvertScript(BaseScript):
    usage = Usage('''
        Convert wake word model from Keras to TensorFlow

        :model str
            Input Keras model (.net)

        :-o --out str {model}.tflite
            Custom output TensorFlow Lite filename
    ''')

    def run(self):
        args = self.args
        model_name = args.model.replace('.net', '')
        self.convert(args.model, args.out.format(model=model_name))

    def convert(self, model_path: str, out_file: str):
        """
        Converts an HD5F file from Keras to a .tflite for use with TensorFlow Runtime

        Args:
            model_path: location of Keras model
            out_file: location to write TFLite model
        """
        print('Converting', model_path, 'to', out_file, '...')

        import tensorflow as tf # Using tensorflow v2.2
        from tensorflow import keras as K
        from precise.model import load_precise_model
        from precise.functions import weighted_log_loss

        out_dir, filename = split(out_file)
        out_dir = out_dir or '.'
        os.makedirs(out_dir, exist_ok=True)

        # Load custom loss function with model
        model = K.models.load_model(model_path, custom_objects={'weighted_log_loss': weighted_log_loss})
        model.summary()

        # Support for freezing Keras models to .pb has been removed in TF 2.0.

        # Converting instead to TFLite model
        print('Starting TFLite conversion.')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        open(out_file, "wb").write(tflite_model)
        print('Wrote to ' + out_file)


main = ConvertScript.run_main

if __name__ == '__main__':
    main()
