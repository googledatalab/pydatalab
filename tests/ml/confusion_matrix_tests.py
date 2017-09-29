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

from google.datalab.ml import ConfusionMatrix


class TestConfusionMatrix(unittest.TestCase):
  """google.datalab.ml.ConfusionMatrix Tests"""

  def setUp(self):
    self._test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self._test_dir)

  def test_from_csv(self):
    """test ConfusionMatrix's from_csv()."""

    counts = self._create_csv_files(['1.csv', '2.csv'], 50)
    cm = ConfusionMatrix.from_csv(os.path.join(self._test_dir, '*.csv'),
                                  headers=['target', 'predicted', ''])
    df = cm.to_dataframe()
    dict_from_df = {}
    for index, row in df.iterrows():
      dict_from_df[(row['target'], row['predicted'])] = row['count']
    self.assertEqual(counts, dict_from_df)

  def _create_csv_files(self, filenames, num_lines):
    """Makes csv data files."""

    counts = {
        ('red', 'red'): 0,
        ('red', 'blue'): 0,
        ('red', 'green'): 0,
        ('blue', 'red'): 0,
        ('blue', 'blue'): 0,
        ('blue', 'green'): 0,
        ('green', 'red'): 0,
        ('green', 'blue'): 0,
        ('green', 'green'): 0
    }
    index = 0
    for filename in filenames:
      full_file_name = os.path.join(self._test_dir, filename)
      with open(full_file_name, 'w') as f:
        writer = csv.writer(f)
        for r in range(num_lines):
          target = random.choice(['red', 'blue', 'green'])
          predicted = random.choice(['red', 'blue', 'green'])
          counts[(target, predicted)] += 1
          index += 1
          writer.writerow([target, predicted, index])

    return counts
