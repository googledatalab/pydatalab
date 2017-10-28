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
import pandas as pd
import random
import shutil
import tempfile

import google.datalab
from google.datalab.ml import CsvDataSet, BigQueryDataSet, TransformedDataSet


class TestCsvDataSet(unittest.TestCase):

  def test_schema(self):
    json_schema = TestCsvDataSet._create_json_schema()

    # CsvDataSet can take a json schema, a Schema object, or a string
    ds = CsvDataSet(file_pattern='some/file', schema=json_schema)
    self.assertEqual(json_schema, ds.schema)

    schema_obj = google.datalab.bigquery.Schema(json_schema)
    ds = CsvDataSet(file_pattern='some/file', schema=schema_obj)
    self.assertEqual(json_schema, ds.schema)

    schema_str = 'id: INTEGER, field1: STRING, field2: INTEGER'
    ds = CsvDataSet(file_pattern='some/file', schema=schema_str)
    self.assertEqual(json_schema, ds.schema)

  def test_sample_and_size(self):
    tmp_dir = tempfile.mkdtemp()
    try:
      json_schema = TestCsvDataSet._create_json_schema()
      all_rows = TestCsvDataSet._create_csv_files(tmp_dir, 'data', 3, 30)
      ds = CsvDataSet(file_pattern=os.path.join(tmp_dir, 'data*'), schema=json_schema)
      self.assertEqual(90, ds.size)

      df = ds.sample(5)
      self.assertEqual(5, len(df))
      self.assertTrue(isinstance(df, pd.DataFrame))
      self.assertEqual(5, len(set(df['id'].tolist())))  # 5 unique rows.

      # check the 5 samples below to the csv files by checking they are in
      # all_rows
      for _, row in df.iterrows():
        row_index = row['id']
        self.assertEqual(all_rows.iloc[row_index]['field1'], row['field1'])
        self.assertEqual(all_rows.iloc[row_index]['field2'], row['field2'])

      df = ds.sample(3 * 5)
      self.assertEqual(3 * 5, len(df))
      self.assertTrue(isinstance(df, pd.DataFrame))
      self.assertEqual(3 * 5, len(set(df['id'].tolist())))  # 15 unique rows.

      with self.assertRaises(ValueError):
        df = ds.sample(3 * 50 + 1)  # sample is larger than data size
    finally:
      shutil.rmtree(tmp_dir)

  @staticmethod
  def _create_csv_files(folder, filename, num_files, num_lines):
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
      num_lines: how many lines each file includes

    Returns:
      A pandas dataframe containing all the csv rows where the id is the index
          row.
    """
    ex_id = 0
    dfs = []
    for i in range(1, num_files + 1):
      full_file_name = os.path.join(folder, filename + str(i) + '.csv')
      with open(full_file_name, 'w') as f:
        writer = csv.writer(f)
        for r in range(num_lines):
          writer.writerow([ex_id,
                           random.choice(['red', 'blue', 'green']),
                           random.randint(0, 100)])
          ex_id += 1
      dfs.append(pd.read_csv(
          full_file_name,
          names=['id', 'field1', 'field2'],
          index_col='id',
          header=None))
    return pd.concat(dfs, axis=0, ignore_index=False)

  @staticmethod
  def _create_json_schema():
    return [{'name': 'id', 'type': 'INTEGER'},      # unique id
            {'name': 'field1', 'type': 'STRING'},   # random string
            {'name': 'field2', 'type': 'INTEGER'}]  # random int


class TestBigQueryDataSet(unittest.TestCase):

  def test_basics(self):
    # Just run the init function. Expand when we have credentials in tests
    BigQueryDataSet(table='a.b')
    BigQueryDataSet(sql='SELECT * FROM myds.mytable')


class TestTransformedDataSet(unittest.TestCase):

  def test_basics(self):
    # Just run the init function.
    TransformedDataSet('a*.gz')
    TransformedDataSet(['a.gz', 'b.gz'])
