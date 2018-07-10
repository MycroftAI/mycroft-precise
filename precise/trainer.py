from argparse import ArgumentParser
from fitipy import Fitipy
from keras.callbacks import LambdaCallback
from os.path import splitext
from prettyparse import add_to_parser
from typing import Any, Tuple

from precise.functions import set_loss_bias
from precise.model import create_model
from precise.params import inject_params, save_params
from precise.train_data import TrainData


class Trainer:
    usage = '''
        Train a new model on a dataset
        
        :model str
            Keras model file (.net) to load from and save to
        
        :-e --epochs int 10
            Number of epochs to train model for
        
        :-s --sensitivity float 0.2
            Weighted loss bias. Higher values decrease increase positives
        
        :-b --batch-size int 5000
            Batch size for training
        
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

    def __init__(self, parser=None):
        parser = parser or ArgumentParser()
        add_to_parser(parser, self.usage, True)
        self.args = args = TrainData.parse_args(parser)
        if not 0.0 <= args.sensitivity <= 1.0:
            parser.error('sensitivity must be between 0.0 and 1.0')

        inject_params(args.model)
        save_params(args.model)
        self.train, self.test = self.load_data(self.args)

        set_loss_bias(1.0 - args.sensitivity)
        self.model = create_model(args.model, args.no_validation, args.extra_metrics)
        self.model.summary()

        from keras.callbacks import ModelCheckpoint, TensorBoard
        checkpoint = ModelCheckpoint(args.model, monitor=args.metric_monitor,
                                     save_best_only=args.save_best)
        epoch_fiti = Fitipy(splitext(args.model)[0] + '.epoch')
        self.epoch = epoch_fiti.read().read(0, int)

        def on_epoch_end(a, b):
            self.epoch += 1
            epoch_fiti.write().write(self.epoch, str)

        self.callbacks = [
            checkpoint, TensorBoard(),
            LambdaCallback(on_epoch_end=on_epoch_end)
        ]

    @staticmethod
    def load_data(args: Any) -> Tuple[tuple, tuple]:
        data = TrainData.from_both(args.tags_file, args.tags_folder, args.folder)
        print('Data:', data)
        train, test = data.load(True, not args.no_validation)

        print('Inputs shape:', train[0].shape)
        print('Outputs shape:', train[1].shape)

        if test:
            print('Test inputs shape:', test[0].shape)
            print('Test outputs shape:', test[1].shape)

        if 0 in train[0].shape or 0 in train[1].shape:
            print('Not enough data to train')
            exit(1)

        return train, test

    def run(self):
        try:
            self.model.fit(
                self.train[0], self.train[1], self.args.batch_size, self.epoch + self.args.epochs,
                validation_data=self.test, initial_epoch=self.epoch,
                callbacks=self.callbacks
            )
        except KeyboardInterrupt:
            print()
