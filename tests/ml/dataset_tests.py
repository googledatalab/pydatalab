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

import google.datalab
from google.datalab.ml import CsvDataSet


class TestCsvDataSet(unittest.TestCase):

  def test_schema(self):
    json_schema = TestCsvDataSet._create_json_schema()

    # CsvDataSet can take a json schema, a Schema object, or a string
    ds = CsvDataSet(file_pattern='some/file', schema=json_schema)
    self.assertEqual(json_schema, ds.schema)

    schema_obj = google.datalab.bigquery.Schema(json_schema)
    ds = CsvDataSet(file_pattern='some/file', schema=schema_obj)
    self.assertEqual(json_schema, ds.schema)

    schema_str = 'field1: STRING, field2: INTEGER'
    ds = CsvDataSet(file_pattern='some/file', schema=schema_str)
    self.assertEqual(json_schema, ds.schema)

  def test_sample(self):
    tmp_dir = tempfile.mkdtemp()
    try:
      json_schema = TestCsvDataSet._create_json_schema()
      TestCsvDataSet._create_csv_files(tmp_dir, 'data', 3)
      ds = CsvDataSet(file_pattern=os.path.join(tmp_dir, 'data*'), schema=json_schema)

      df = ds.sample(5)
      self.assertEqual(5, len(df))

      df = ds.sample(3 * 5)
      self.assertEqual(3 * 5, len(df))

      with self.assertRaises(ValueError):
        df = ds.sample(3 * 5 + 1)  # sample is larger than data size
    finally:
      shutil.rmtree(tmp_dir)

  @staticmethod
  def _create_csv_files(folder, filename, num_files):
    """Makes csv data files.

    Makes files in the from:
      folder/filename1.csv,
      folder/filename2.csv,
      ...
      folder/filename{num_files}.csv

    Each file will have 5 random csv rows.

    Args:
      folder: output folder
      filename: filename prefix
      num_files: how many files to make
    """
    for i in range(1, num_files + 1):
      with open(os.path.join(folder, filename + str(i) + '.csv'), 'w') as f:
        writer = csv.writer(f)
        for r in range(5):
          writer.writerow([random.choice(['red', 'blue', 'green']),
                           random.randint(0, 100)])

  @staticmethod
  def _create_json_schema():
    return [{'name': 'field1', 'type': 'STRING'},
            {'name': 'field2', 'type': 'INTEGER'}]
