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

import h5py
import numpy
# Optimizer blackhat
from bbopt import BlackBoxOptimizer
from decimal import *
from keras.layers.core import Dense
from keras.layers.recurrent import GRU
from keras.models import Sequential
from pprint import pprint
from typing import *

from precise.functions import weighted_log_loss
from precise.params import pr
from precise.train_data import TrainData


def false_pos(yt, yp) -> Any:
    from keras import backend as K
    return K.sum(K.cast(yp * (1 - yt) > 0.5, 'float')) / K.sum(1 - yt)


def false_neg(yt, yp) -> Any:
    from keras import backend as K
    return K.sum(K.cast((1 - yp) * (0 + yt) > 0.5, 'float')) / K.sum(0 + yt)


def main():
    bb = BlackBoxOptimizer(file=__file__)

    # Loading in data to train
    data = TrainData.from_both('/home/mikhail/wakewords/wakewords/files/tags.txt',
                               '/home/mikhail/wakewords/wakewords/files',
                               '/home/mikhail/wakewords/wakewords/not-wake-word/generated')
    (train_inputs, train_outputs), (test_inputs, test_outputs) = data.load()

    test_data = (test_inputs, test_outputs)

    for i in range(5):
        bb.run(backend="random")

        print("\n= %d = (example #%d)" % (i + 1, len(bb.get_data()["examples"]) + 1))

        shuffle_ids = numpy.arange(len(test_inputs))
        numpy.random.shuffle(shuffle_ids)
        (test_inputs, test_outputs) = (test_inputs[shuffle_ids], test_outputs[shuffle_ids])

        model_array = numpy.empty(len(test_data), dtype=int)
        with h5py.File('tested_models.hdf5', 'w') as f:
            f.create_dataset('dataset_1', data=model_array)
        f.close()

        batch_size = bb.randint("batch_size", 1000, 5000, guess=3000)

        model = Sequential()
        model.add(GRU(units=bb.randint("units", 1, 100, guess=50), activation='linear',
                      input_shape=(pr.n_features, pr.feature_size),
                      dropout=bb.uniform("dropout", 0.1, 0.9, guess=0.6), name='net'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile('rmsprop', weighted_log_loss, metrics=['accuracy'])

        # goodness metric for optimization
        def goodness(y_true, y_pred) -> Any:
            from math import exp
            try:
                param_score = 1.0 / (1.0 + exp((model.count_params() - 11000) / 2000))
            except OverflowError:
                param_score = 1.0 / (1.0 + Decimal(exp((model.count_params() - 11000)) / 2000))
            fitness = param_score * (
                ((1.0 - (0.05 * false_neg(y_true, y_pred))) - (0.95 * false_pos(y_true, y_pred))))
            return fitness

        from keras.callbacks import ModelCheckpoint

        checkpoint = ModelCheckpoint('tested_models.hdf5', monitor='val_loss',
                                     save_best_only=True)

        train_history = model.fit(train_inputs, train_outputs, batch_size=batch_size, epochs=100,
                                  validation_data=(test_inputs, test_outputs),
                                  callbacks=[checkpoint])
        test_loss, test_acc = model.evaluate(test_inputs, test_outputs)

        predictions = model.predict(test_inputs)
        num_false_positive = numpy.sum(predictions * (1 - test_outputs) > 0.5)
        num_false_negative = numpy.sum((1 - predictions) * test_outputs > 0.5)
        false_positives = num_false_positive / numpy.sum(test_outputs < 0.5)
        false_negatives = num_false_negative / numpy.sum(test_outputs > 0.5)

        bb.remember({
            "test loss": test_loss,
            "test accuracy": test_acc,
            "false positive%": false_positives,
            "false negative%": false_negatives
        })
        print(false_positives)
        print("False positive: ", false_positives * 100, "%")
        bb.minimize(false_positives)
        pprint(bb.get_current_run())

    best_example = bb.get_optimal_run()
    print("\n= BEST = (example #%d)" % bb.get_data()["examples"].index(best_example))
    pprint(best_example)


if __name__ == '__main__':
    main()
