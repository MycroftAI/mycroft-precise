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
{accuracy_ratio:.2f}

{false_pos_ratio:.2%} false positives
{false_neg_ratio:.2%} false negatives
'''


class Stats:
    """Represents a set of statistics from a model run on a dataset"""
    def __init__(self, outputs, targets, filenames):
        self.outputs = outputs
        self.targets = targets
        self.filenames = filenames
        self.num_positives = sum(int(i > 0.5) for i in self.targets)
        self.num_negatives = sum(int(i < 0.5) for i in self.targets)

        # Methods
        self.false_positives = lambda threshold=0.5: self.calc_metric(False, True, threshold) / self.num_negatives
        self.false_negatives = lambda threshold=0.5: self.calc_metric(False, False, threshold) / self.num_positives
        self.num_correct = lambda threshold=0.5: sum(
            int(output >= threshold) == int(target)
            for output, target in zip(self.outputs, self.targets)
        )
        self.num_incorrect = lambda threshold=0.5: len(self) - self.num_correct(threshold)
        self.accuracy = lambda threshold=0.5: self.num_correct(threshold) / len(self)

    def __len__(self):
        return len(self.outputs)

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
            if self.matches_sample(output, target, threshold, is_correct, actual_output)
        ]

    def calc_metric(self, is_correct: bool, actual_output: bool, threshold=0.5) -> int:
        """For example, calc_metric(False, True) calculates false positives"""
        return sum(
            self.matches_sample(output, target, threshold, is_correct, actual_output)
            for output, target, filename in zip(self.outputs, self.targets, self.filenames)
        )

    @staticmethod
    def matches_sample(output, target, threshold, is_correct, actual_output):
        """
        Check if a sample with the given network output, target output, and threshold
        is the classification (is_correct, actual_output) like true positive or false negative
        """
        return (bool(output > threshold) == bool(target)) == is_correct and actual_output == bool(output > threshold)
