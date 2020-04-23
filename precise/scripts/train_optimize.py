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
Use black box optimization to tune model hyperparameters. Call
this script in a loop to iteratively tune parameters

:trials_name str
    Filename to save hyperparameter optimization trials in
    '.bbopt.json' will automatically be appended

:noise_folder str
    Folder with random noise to evaluate ambient activations

:-ie --interaction-estimate int 100
    Estimated number of interactions per day

:-aaa --ambient-activation-annoyance float 1.0
    An ambient activation is X times as annoying as a failed
    activation when the wake word is spoken

:-bp --base-params str {}
    Json string containing base ListenerParams for all models

...
"""
import json
from math import exp
from uuid import uuid4

from keras.models import save_model
from prettyparse import Usage

from precise.annoyance_estimator import AnnoyanceEstimator
from precise.model import ModelParams, create_model
from precise.params import pr, save_params
from precise.scripts.train import TrainScript
from precise.stats import Stats


class TrainOptimizeScript(TrainScript):
    usage = Usage(__doc__) | TrainScript.usage
    del usage.arguments['model']  # Remove 'model' argument from original TrainScript

    def __init__(self, args):
        from bbopt import BlackBoxOptimizer
        pr.__dict__.update(json.loads(args.base_params))
        args.model = args.trials_name + '-cur'
        save_params(args.model)
        super().__init__(args)
        self.bb = BlackBoxOptimizer(file=self.args.trials_name)

    def calc_params_cost(self, model):
        """
        Models the real world cost of additional model parameters
        Up to a certain point, having more parameters isn't worse.
        However, at a certain point more parameters will risk
        running slower than realtime and become unfeasible. This
        is why it's modelled exponentially with some reasonable
        number of acceptable parameters.

        Ideally, this would be replaced with floating point
        computations and the numbers would be configurable
        rather than chosen relatively arbitrarily
        """
        return 1.0 + exp((model.count_params() - 11000) / 10000)

    def run(self):
        self.bb.run(alg='tree_structured_parzen_estimator')

        model = create_model(None, ModelParams(
            recurrent_units=self.bb.randint("units", 1, 120, guess=30),
            dropout=self.bb.uniform("dropout", 0.05, 0.9, guess=0.2),
            extra_metrics=self.args.extra_metrics,
            skip_acc=self.args.no_validation,
            loss_bias=self.bb.uniform(
                'loss_bias', 0.01, 0.99, guess=1.0 - self.args.sensitivity
            ),
            freeze_till=0
        ))
        model.fit(
            *self.sampled_data, batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            validation_data=self.test * (not self.args.no_validation),
            callbacks=[]
        )
        test_in, test_out = self.test
        test_pred = model.predict(test_in, batch_size=self.args.batch_size)
        stats_dict = Stats(test_pred, test_out, []).to_dict()

        ann_est = AnnoyanceEstimator(
            model, self.args.interaction_estimate,
            self.args.ambient_activation_annoyance
        ).estimate(
            model, test_pred, test_out,
            self.args.noise_folder, self.args.batch_size
        )
        params_cost = self.calc_params_cost(model)
        cost = ann_est.annoyance + params_cost

        model_name = '{}-{}.net'.format(self.args.trials_name, str(uuid4()))
        save_model(model, model_name)
        save_params(model_name)

        self.bb.remember({
            'test_stats': stats_dict,
            'best_threshold': ann_est.threshold,
            'cost': cost,
            'cost_info': {
                'params_cost': params_cost,
                'annoyance': ann_est.annoyance,
                'ww_annoyance': ann_est.ww_annoyance,
                'nww_annoyance': ann_est.nww_annoyance,
            },
            'model': model_name
        })
        print('Current Run: {}'.format(json.dumps(
            self.bb.get_current_run(), indent=4
        )))
        self.bb.minimize(cost)


main = TrainOptimizeScript.run_main

if __name__ == '__main__':
    main()
