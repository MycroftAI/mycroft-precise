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
# limitations under the License
"""
Module handles computing and formatting basic statistics
about a dataset like false negatives and false positives
"""
import numpy as np

counts_str = '''
=== Counts ===
False Positives: {false_pos}
True Negatives: {true_neg}
False Negatives: {false_neg}
True Positives: {true_pos}
'''.strip()

summary_str = '''
=== Summary ===
{num_correct} out of {total}
{accuracy_ratio:.2%}

{false_pos_ratio:.2%} false positives
{false_neg_ratio:.2%} false negatives
'''


class Stats:
    """Represents a set of statistics from a model run on a dataset"""

    def __init__(self, outputs, targets, filenames):
        self.outputs = np.array(outputs)
        self.targets = np.array(targets)
        self.filenames = filenames
        self.num_positives = int((self.targets > 0.5).sum())
        self.num_negatives = int((self.targets < 0.5).sum())

        # Methods
        self.false_positives = lambda threshold=0.5: self.calc_metric(False, True, threshold) / max(1,
                                                                                                    self.num_negatives)
        self.false_negatives = lambda threshold=0.5: self.calc_metric(False, False, threshold) / max(1,
                                                                                                     self.num_positives)
        self.num_correct = lambda threshold=0.5: (
                (self.outputs >= threshold) == self.targets.astype(bool)
        ).sum()
        self.num_incorrect = lambda threshold=0.5: len(self) - self.num_correct(threshold)
        self.accuracy = lambda threshold=0.5: self.num_correct(threshold) / max(1, len(self))

    def __len__(self):
        return len(self.outputs)

    def to_np_dict(self):
        import numpy as np
        return {
            'outputs': self.outputs,
            'targets': self.targets,
            'filenames': np.array(self.filenames)
        }

    @staticmethod
    def from_np_dict(data) -> 'Stats':
        return Stats(data['outputs'], data['targets'], data['filenames'])

    def to_dict(self, threshold=0.5):
        return {
            'true_pos': self.calc_metric(True, True, threshold),
            'true_neg': self.calc_metric(True, False, threshold),
            'false_pos': self.calc_metric(False, True, threshold),
            'false_neg': self.calc_metric(False, False, threshold),
        }

    def counts_str(self, threshold=0.5):
        return counts_str.format(**self.to_dict(threshold))

    def summary_str(self, threshold=0.5):
        return summary_str.format(
            num_correct=self.num_correct(threshold), total=len(self),
            accuracy_ratio=self.accuracy(threshold),
            false_pos_ratio=self.false_positives(threshold),
            false_neg_ratio=self.false_negatives(threshold)
        )

    def calc_filenames(self, is_correct: bool, actual_output: bool, threshold=0.5) -> list:
        """Find a list of files with the given classification"""
        return [
            filename
            for output, target, filename in zip(self.outputs, self.targets, self.filenames)
            if ((output > threshold) == bool(target)) == is_correct and actual_output == bool(output > threshold)
        ]

    def calc_metric(self, is_correct: bool, actual_output: bool, threshold=0.5) -> int:
        """For example, calc_metric(False, True) calculates false positives"""
        return int(
            ((((self.outputs > threshold) == self.targets.astype(bool)) == is_correct) &
             ((self.outputs > threshold) == actual_output)).sum()
        )

    @staticmethod
    def matches_sample(output, target, threshold, is_correct, actual_output):
        """
        Check if a sample with the given network output, target output, and threshold
        is the classification (is_correct, actual_output) like true positive or false negative
        """
        return (bool(output > threshold) == bool(target)) == is_correct and actual_output == bool(output > threshold)
