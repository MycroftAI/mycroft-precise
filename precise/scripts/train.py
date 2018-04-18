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
from prettyparse import create_parser

from precise.model import create_model
from precise.params import inject_params, save_params
from precise.train_data import TrainData

usage = '''
    Train a new model on a dataset
    
    :model str
        Keras model file (.net) to load from and save to
    
    :-e --epochs int 10
        Number of epochs to train model for
    
    :-sb --save-best
        Only save the model each epoch if its stats improve
    
    :-nv --no-validation
        Disable accuracy and validation calculation
        to improve speed during training
    
    :-mm --metric-monitor str loss
        Metric used to determine when to save
    
    :-em --extra-metrics
        Add extra metrics during training
    
    ...
'''


def main():
    args = TrainData.parse_args(create_parser(usage))

    inject_params(args.model)
    save_params(args.model)

    data = TrainData.from_both(args.tags_file, args.tags_folder, args.folder)
    print('Data:', data)
    (inputs, outputs), test_data = data.load(True, not args.no_validation)

    print('Inputs shape:', inputs.shape)
    print('Outputs shape:', outputs.shape)

    if test_data:
        print('Test inputs shape:', test_data[0].shape)
        print('Test outputs shape:', test_data[1].shape)

    if 0 in inputs.shape or 0 in outputs.shape:
        print('Not enough data to train')
        exit(1)

    model = create_model(args.model, args.no_validation, args.extra_metrics)
    model.summary()

    from keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(args.model, monitor=args.metric_monitor,
                                 save_best_only=args.save_best)

    try:
        model.fit(inputs, outputs, 5000, args.epochs, validation_data=test_data,
                  callbacks=[checkpoint])
    except KeyboardInterrupt:
        print()
    finally:
        model.save(args.model)


if __name__ == '__main__':
    main()
