# Copyright 2020 Mycroft AI Inc.
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
from collections import namedtuple
from glob import glob
from os.path import join

import numpy as np

from precise.params import pr
from precise.util import load_audio
from precise.vectorization import vectorize_raw

AnnoyanceEstimate = namedtuple(
    'AnnoyanceEstimate',
    'annoyance ww_annoyance nww_annoyance threshold'
)


class AnnoyanceEstimator:
    """
    This class attempts to estimate the "annoyance" of a user
    of a given network. It models annoyance as follows:

    Annoyance from false negatives (not activating when it should):
    We assume that the annoyance incurred by each subsequent failed
    activation attempt is double that of the previous attempt. ie.
    two failed activations causes 1 + 2 = 3 annoyance units but three
    failed activations causes 1 + 2 + 4 = 7 annoyance units.

    Annoyance from false positives (activating when it should not):
    We assume that each false positive incurs some constant annoyance

    With this, we can compute net annoyance from false positives
    and negatives individually, combine them for the total annoyance.

    Finally, we can recompute this annoyance for each threshold
    value to find the threshold that yields the lowest net annoyance
    """

    def __init__(self, model, interaction_estimate, ambient_annoyance):
        self.thresholds = 1 / (1 + np.exp(-np.linspace(-20, 20, 1000)))
        self.interaction_estimate = interaction_estimate
        self.ambient_annoyance = ambient_annoyance

    def compute_nww_annoyances(self, model, noise_folder, batch_size):
        """
        Given some number, x, of ambient activations per hour, we can
        compute the annoyance per day from false positives as 24 * x
        times the annoyance incurred per ambient activation.
        """
        nww_seconds = 0.0
        nww_buckets = np.zeros_like(self.thresholds)
        for i in glob(join(noise_folder, '*.wav')):
            print('Evaluating ambient activations on {}...'.format(i))
            inputs, audio_len = self._load_inputs(i)
            nww_seconds += audio_len / pr.sample_rate
            ambient_predictions = model.predict(inputs, batch_size=batch_size)
            del inputs
            nww_buckets += (ambient_predictions.reshape((-1, 1))
                            > self.thresholds.reshape((1, -1))).sum(axis=0)
        nww_acts_per_hour = nww_buckets * 60 * 60 / nww_seconds
        return self.ambient_annoyance * nww_acts_per_hour * 24

    def compute_ww_annoyances(self, ww_predictions):
        """
        Given some proportion, p, of not recognizing the wake word, our
        total annoyance per interaction is modelled as p^1 * 2^0 + p^2 * 2^1
        + ... + p^i * 2^(i - 1) which converges to 1 / (1 - 2p) - 1.
        Given some number of interactions per day we can then find the
        expected annoyance per day from false negatives.
        """
        ww_buckets = (ww_predictions.reshape((-1, 1)) >
                      self.thresholds.reshape((1, -1))).sum(axis=0)
        ww_fail_ratios = 1 - ww_buckets / len(ww_predictions)
        # Performs 1 / (1 - 2 * ww_fail_ratios) - 1, handling edge case
        ann_per_interaction = np.divide(
            1, 1 - 2 * ww_fail_ratios,
            where=ww_fail_ratios < 0.5
        ) - 1
        ann_per_interaction[ww_fail_ratios >= 0.5] = float('inf')
        return self.interaction_estimate * ann_per_interaction

    def estimate(self, model, predictions, targets, noise_folder, batch_size):
        """
        Estimates the annoyance a model incurs according to the model
        described in the class documentation
        """
        ww_predictions = predictions[np.where(targets > 0.5)]
        ww_annoyances = self.compute_ww_annoyances(ww_predictions)
        nww_annoyances = self.compute_nww_annoyances(
            model, noise_folder, batch_size
        )
        annoyance_by_threshold = ww_annoyances + nww_annoyances
        best_threshold_id = np.argmin(annoyance_by_threshold)
        min_annoyance = annoyance_by_threshold[best_threshold_id]
        return AnnoyanceEstimate(
            annoyance=min_annoyance,
            ww_annoyance=ww_annoyances[best_threshold_id],
            nww_annoyance=nww_annoyances[best_threshold_id],
            threshold=self.thresholds[best_threshold_id]
        )

    @staticmethod
    def _load_inputs(audio_file, chunk_size=4096):
        """
        Loads network inputs from an audio file without caching
        Handles data conservatively in case the audio file is large
        Args:
            audio_file: Filename to load
            chunk_size: Samples to skip forward when loading network inpus
        """
        audio = load_audio(audio_file)
        audio_len = len(audio)
        mfccs = vectorize_raw(audio)
        del audio
        mfcc_hops = chunk_size // pr.hop_samples
        return np.array([
            mfccs[i - pr.n_features:i] for i in range(pr.n_features, len(mfccs), mfcc_hops)
        ]), audio_len
