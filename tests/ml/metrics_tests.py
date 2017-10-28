# Copyright 2017 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

from __future__ import absolute_import
from __future__ import unicode_literals
import unittest

import csv
import os
import random
import shutil
import tempfile

from google.datalab.ml import Metrics


class TestMetrics(unittest.TestCase):
  """google.datalab.ml.Metrics Tests."""

  def setUp(self):
    self._test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self._test_dir)

  def test_accuracy(self):
    """test Metrics's accuracy()."""

    dict_counts, dict_accuracy = self._create_classification_csv_files(
        ['classification1.csv', 'classification2.csv'], 50)
    metrics = Metrics.from_csv(os.path.join(self._test_dir, 'classification?.csv'),
                               headers=['target', 'predicted', ''])
    accuracy = metrics.accuracy()
    dict_counts_from_results, dict_accuracy_from_results = {}, {}
    for index, row in accuracy.iterrows():
      dict_counts_from_results[row['target']] = row['count']
      dict_accuracy_from_results[row['target']] = row['accuracy']

    self.assertEqual(dict_counts, dict_counts_from_results)
    self.assertEqual(dict_accuracy, dict_accuracy_from_results)

  def test_precision_recall(self):
    """test Metrics's accuracy()."""

    self._create_classification_csv_files_with_probs('color.csv', 500)
    metrics = Metrics.from_csv(os.path.join(self._test_dir, 'color.csv'),
                               headers=['target', 'blue', 'key'])
    pr = metrics.precision_recall(10, 'blue')
    self.assertEqual(10, len(pr))
    # When threshold = 0.0 recall should be 1.0
    self.assertAlmostEqual(1.0, pr[pr['threshold'] == 0.0]['recall'][0])
    # Recall values should be desc ordered.
    for i in range(9):
      self.assertAlmostEqual(float(i) / 10, pr['threshold'][i])
      self.assertGreater(pr['recall'][i], pr['recall'][i + 1])

  def test_roc(self):
    """test Metrics's roc()."""

    self._create_classification_csv_files_with_probs('color.csv', 500)
    metrics = Metrics.from_csv(os.path.join(self._test_dir, 'color.csv'),
                               headers=['target', 'blue', 'key'])
    roc = metrics.roc(10, 'blue')
    # note that roc includes both threshold 0.0 and 1.0.
    self.assertEqual(11, len(roc))

    # When threshold = 0.0 fpr and tpr should be 1.0
    self.assertAlmostEqual(1.0, roc[roc['threshold'] == 0.0]['fpr'][0])
    self.assertAlmostEqual(1.0, roc[roc['threshold'] == 0.0]['tpr'][0])

    # When threshold = 1.0 fpr and tpr should be 0.0
    self.assertAlmostEqual(0.0, roc[roc['threshold'] == 1.0]['fpr'][10])
    self.assertAlmostEqual(0.0, roc[roc['threshold'] == 1.0]['tpr'][10])

  def test_rmse(self):
    """test Metrics's accuracy()."""

    truth = self._create_regression_csv_file()
    metrics = Metrics.from_csv(os.path.join(self._test_dir, 'regression.csv'),
                               headers=['key', 'target', 'predicted'])
    rmse = metrics.rmse()
    self.assertEqual(truth['rmse'], rmse)

  def test_mae(self):
    """test Metrics's accuracy()."""

    truth = self._create_regression_csv_file()
    metrics = Metrics.from_csv(os.path.join(self._test_dir, 'regression.csv'),
                               headers=['key', 'target', 'predicted'])
    mae = metrics.mae()
    self.assertEqual(truth['mae'], mae)

  def test_percentile(self):
    """test Metrics's accuracy()."""

    truth = self._create_regression_csv_file()
    metrics = Metrics.from_csv(os.path.join(self._test_dir, 'regression.csv'),
                               headers=['key', 'target', 'predicted'])
    percentile50 = metrics.percentile_nearest(50)
    self.assertEqual(truth['percentile50'], percentile50)
    percentile90 = metrics.percentile_nearest(90)
    self.assertEqual(truth['percentile90'], percentile90)

  def _create_classification_csv_files(self, filenames, num_lines):
    """Makes classification csv data files."""

    dict_counts = {
        'red': 0,
        'blue': 0,
        'green': 0,
        '_all': 0,
    }
    dict_correct_counts = {
        'red': 0,
        'blue': 0,
        'green': 0,
        '_all': 0,
    }
    index = 0
    for filename in filenames:
      full_file_name = os.path.join(self._test_dir, filename)
      with open(full_file_name, 'w') as f:
        writer = csv.writer(f)
        for r in range(num_lines):
          target = random.choice(['red', 'blue', 'green'])
          predicted = random.choice(['red', 'blue', 'green'])
          dict_counts[target] += 1
          dict_counts['_all'] += 1
          if target == predicted:
            dict_correct_counts[target] += 1
            dict_correct_counts['_all'] += 1
          index += 1
          writer.writerow([target, predicted, index])

    dict_accuracy = {
        'red': float(dict_correct_counts['red']) / dict_counts['red'],
        'blue': float(dict_correct_counts['blue']) / dict_counts['blue'],
        'green': float(dict_correct_counts['green']) / dict_counts['green'],
        '_all': float(dict_correct_counts['_all']) / dict_counts['_all'],
    }
    return dict_counts, dict_accuracy

  def _create_classification_csv_files_with_probs(self, filename, num_lines):
    """Makes classification csv data files."""

    index = 0
    full_file_name = os.path.join(self._test_dir, filename)
    with open(full_file_name, 'w') as f:
      writer = csv.writer(f)
      for r in range(num_lines):
        target = random.choice(['red', 'blue', 'green'])
        prob = random.uniform(0, 1)
        writer.writerow([target, prob, index])

  def _create_regression_csv_file(self):
    """Makes a single regression csv data file."""

    full_file_name = os.path.join(self._test_dir, 'regression.csv')
    with open(full_file_name, 'w') as f:
      writer = csv.writer(f)
      writer.writerow([1, 12.4, 15.4])
      writer.writerow([2, 20.1, 16.1])
      writer.writerow([3, 10.0, 10.0])
      writer.writerow([4, 0, 0])

    # rmse = sqrt(((12.4-15.4)^2 + (20.1-16.1)^2 + 0 + 0) / 4) = 2.5
    # mae = ((15.4-12.4) + (20.1-16.1) + 0 + 0) / 4 = 1.75
    return {'rmse': 2.5, 'mae': 1.75, 'percentile50': 3.0, 'percentile90': 4.0}
