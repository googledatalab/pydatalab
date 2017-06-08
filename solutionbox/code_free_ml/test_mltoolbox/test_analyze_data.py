from __future__ import absolute_import
from __future__ import print_function

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import unittest
import pandas as pd
import six

from tensorflow.python.lib.io import file_io

import google.datalab as dl
import google.datalab.bigquery as bq

# To make 'import analyze_data' work without installing it.
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'mltoolbox', 'code_free_ml')))

import analyze_data  # noqa: E303


class TestConfigFiles(unittest.TestCase):
  """Tests for checking the format between the schema and features files."""

  def test_expand_defaults_do_nothing(self):
    schema = [{'name': 'col1', 'type': 'FLOAT'},
              {'name': 'col2', 'type': 'INTEGER'}]
    features = {'col1': {'transform': 'x'},
                'col2': {'transform': 'y'}}
    original_features = copy.deepcopy(features)

    analyze_data.expand_defaults(schema, features)

    # Nothing should change.
    self.assertEqual(original_features, features)

  def test_expand_defaults_unknown_schema_type(self):
    schema = [{'name': 'col1', 'type': 'BYTES'},
              {'name': 'col2', 'type': 'INTEGER'}]
    features = {'col1': {'transform': 'x'},
                'col2': {'transform': 'y'}}

    with self.assertRaises(ValueError):
      analyze_data.expand_defaults(schema, features)

  def test_expand_defaults(self):
    schema = [{'name': 'col1', 'type': 'FLOAT'},
              {'name': 'col2', 'type': 'INTEGER'},
              {'name': 'col3', 'type': 'STRING'},
              {'name': 'col4', 'type': 'FLOAT'},
              {'name': 'col5', 'type': 'INTEGER'},
              {'name': 'col6', 'type': 'STRING'}]
    features = {'col1': {'transform': 'x'},
                'col2': {'transform': 'y'},
                'col3': {'transform': 'z'}}

    analyze_data.expand_defaults(schema, features)

    self.assertEqual(
      features,
      {'col1': {'transform': 'x'},
       'col2': {'transform': 'y'},
       'col3': {'transform': 'z'},
       'col4': {'transform': 'identity'},
       'col5': {'transform': 'identity'},
       'col6': {'transform': 'one_hot'}})

  def test_check_schema_transforms_match(self):
    with self.assertRaises(ValueError):
      analyze_data.check_schema_transforms_match(
         [{'name': 'col1', 'type': 'INTEGER'}],
         {'col1': {'transform': 'one_hot'}})

    with self.assertRaises(ValueError):
      analyze_data.check_schema_transforms_match(
         [{'name': 'col1', 'type': 'FLOAT'}],
         {'col1': {'transform': 'embedding'}})

    with self.assertRaises(ValueError):
      analyze_data.check_schema_transforms_match(
         [{'name': 'col1', 'type': 'STRING'}],
         {'col1': {'transform': 'scale'}})

    with self.assertRaises(ValueError):
      analyze_data.check_schema_transforms_match(
         [{'name': 'col1', 'type': 'xxx'}],
         {'col1': {'transform': 'scale'}})

    with self.assertRaises(ValueError):
      analyze_data.check_schema_transforms_match(
         [{'name': 'col1', 'type': 'INTEGER'}],
         {'col1': {'transform': 'xxx'}})


class TestLocalAnalyze(unittest.TestCase):
  """Test local analyze functions."""

  def test_numerics(self):
    output_folder = tempfile.mkdtemp()
    input_file_path = tempfile.mkstemp(dir=output_folder)[1]
    try:
      file_io.write_string_to_file(
        input_file_path,
        '\n'.join(['%s,%s' % (i, 10 * i + 0.5) for i in range(100)]))

      analyze_data.run_local_analysis(
        output_folder,
        input_file_path,
        [{'name': 'col1', 'type': 'INTEGER'},
         {'name': 'col2', 'type': 'FLOAT'}],
        {'col1': {'transform': 'scale'},
         'col2': {'transform': 'identity'}})
      stats = json.loads(
          file_io.read_file_to_string(
              os.path.join(output_folder, analyze_data.constant.STATS_FILE)).decode())

      self.assertEqual(stats['num_examples'], 100)
      col = stats['column_stats']['col1']
      self.assertAlmostEqual(col['max'], 99.0)
      self.assertAlmostEqual(col['min'], 0.0)
      self.assertAlmostEqual(col['mean'], 49.5)

      col = stats['column_stats']['col2']
      self.assertAlmostEqual(col['max'], 990.5)
      self.assertAlmostEqual(col['min'], 0.5)
      self.assertAlmostEqual(col['mean'], 495.5)
    finally:
      shutil.rmtree(output_folder)

  def test_categorical(self):
    output_folder = tempfile.mkdtemp()
    input_file_path = tempfile.mkstemp(dir=output_folder)[1]
    try:
      csv_file = ['red,car', 'red,truck', 'red,van', 'blue,bike', 'blue,train',
                  'green,airplane']
      file_io.write_string_to_file(
        input_file_path,
        '\n'.join(csv_file))

      analyze_data.run_local_analysis(
        output_folder,
        input_file_path,
        [{'name': 'color', 'type': 'STRING'},
         {'name': 'transport', 'type': 'STRING'}],
        {'color': {'transform': 'one_hot'},
         'transport': {'transform': 'embedding'}})

      stats = json.loads(
          file_io.read_file_to_string(
              os.path.join(output_folder, analyze_data.constant.STATS_FILE)).decode())
      self.assertEqual(stats['column_stats']['color']['vocab_size'], 3)
      self.assertEqual(stats['column_stats']['transport']['vocab_size'], 6)

      # Color column.
      vocab_str = file_io.read_file_to_string(
        os.path.join(output_folder, analyze_data.constant.VOCAB_ANALYSIS_FILE % 'color'))
      vocab = pd.read_csv(six.StringIO(vocab_str),
                          header=None,
                          names=['color', 'count'])
      expected_vocab = pd.DataFrame(
          {'color': ['red', 'blue', 'green'], 'count': [3, 2, 1]},
          columns=['color', 'count'])
      pd.util.testing.assert_frame_equal(vocab, expected_vocab)

      # transport column. As each vocab has the same count, order in file is
      # not known.
      vocab_str = file_io.read_file_to_string(
          os.path.join(output_folder,
                       analyze_data.constant.VOCAB_ANALYSIS_FILE % 'transport'))
      vocab = pd.read_csv(six.StringIO(vocab_str),
                          header=None,
                          names=['transport', 'count'])
      self.assertEqual(vocab['count'].tolist(), [1 for i in range(6)])
      self.assertItemsEqual(vocab['transport'].tolist(),
                            ['car', 'truck', 'van', 'bike', 'train', 'airplane'])
    finally:
      shutil.rmtree(output_folder)

  def test_text(self):
    output_folder = tempfile.mkdtemp()
    input_file_path = tempfile.mkstemp(dir=output_folder)[1]
    try:
      csv_file = ['the quick brown fox,raining in kir',
                  'quick   brown brown chicken,raining in pdx']
      file_io.write_string_to_file(
        input_file_path,
        '\n'.join(csv_file))

      analyze_data.run_local_analysis(
        output_folder,
        input_file_path,
        [{'name': 'col1', 'type': 'STRING'}, {'name': 'col2', 'type': 'STRING'}],
        {'col1': {'transform': 'bag_of_words'},
         'col2': {'transform': 'tfidf'}})

      stats = json.loads(
          file_io.read_file_to_string(
              os.path.join(output_folder, analyze_data.constant.STATS_FILE)).decode())
      self.assertEqual(stats['column_stats']['col1']['vocab_size'], 5)
      self.assertEqual(stats['column_stats']['col2']['vocab_size'], 4)

      vocab_str = file_io.read_file_to_string(
          os.path.join(output_folder,
                       analyze_data.constant.VOCAB_ANALYSIS_FILE % 'col1'))
      vocab = pd.read_csv(six.StringIO(vocab_str),
                          header=None,
                          names=['col1', 'count'])
      self.assertEqual(vocab['col1'].tolist(),
                       ['quick', 'brown', 'the', 'fox', 'chicken'])
      self.assertEqual(vocab['count'].tolist(), [2, 2, 1, 1, 1])

      vocab_str = file_io.read_file_to_string(
          os.path.join(output_folder,
                       analyze_data.constant.VOCAB_ANALYSIS_FILE % 'col2'))
      vocab = pd.read_csv(six.StringIO(vocab_str),
                          header=None,
                          names=['col2', 'count'])
      self.assertEqual(vocab['col2'].tolist(), ['raining', 'in', 'pdx', 'kir'])
      self.assertEqual(vocab['count'].tolist(), [2, 2, 1, 1])
    finally:
      shutil.rmtree(output_folder)


class TestCloudAnalyzeFromBQTable(unittest.TestCase):
  """Test the analyze functions using data in a BigQuery table.

  As the SQL statements do not change if the BigQuery source is csv fiels or a
  real table, there is no need to test every SQL analyze statement. We only run
  one test to make sure this path works.
  """

  def test_numerics(self):
    """Build a BQ table, and then call analyze on it."""
    schema = [{'name': 'col1', 'type': 'INTEGER'},
              {'name': 'col2', 'type': 'FLOAT'}]
    project_id = dl.Context.default().project_id
    dataset_name = 'temp_pydatalab_test_%s' % uuid.uuid4().hex
    table_name = 'temp_table'
    full_table_name = '%s.%s.%s' % (project_id, dataset_name, table_name)

    output_folder = tempfile.mkdtemp()

    try:
      # Make a dataset, a table, and insert data.
      db = bq.Dataset((project_id, dataset_name))
      db.create()

      table = bq.Table(full_table_name)
      table.create(schema=bq.Schema(schema), overwrite=True)

      data = [{'col1': i, 'col2': 10 * i + 0.5} for i in range(100)]
      table.insert(data)

      analyze_data.run_cloud_analysis(
          output_dir=output_folder,
          csv_file_pattern=None,
          bigquery_table=full_table_name,
          schema=schema,
          features={'col1': {'transform': 'scale'},
                    'col2': {'transform': 'identity'}})

      stats = json.loads(
          file_io.read_file_to_string(
              os.path.join(output_folder, analyze_data.constant.STATS_FILE)).decode())

      self.assertEqual(stats['num_examples'], 100)
      col = stats['column_stats']['col1']
      self.assertAlmostEqual(col['max'], 99.0)
      self.assertAlmostEqual(col['min'], 0.0)
      self.assertAlmostEqual(col['mean'], 49.5)

      col = stats['column_stats']['col2']
      self.assertAlmostEqual(col['max'], 990.5)
      self.assertAlmostEqual(col['min'], 0.5)
      self.assertAlmostEqual(col['mean'], 495.5)
    finally:
      shutil.rmtree(output_folder)
      db.delete(delete_contents=True)


class TestCloudAnalyzeFromCSVFiles(unittest.TestCase):
  """Test the analyze function using BigQuery from csv files that are on GCS."""

  @classmethod
  def setUpClass(cls):
    cls._bucket_root = 'gs://temp_pydatalab_test_%s' % uuid.uuid4().hex
    subprocess.check_call('gsutil mb %s' % cls._bucket_root, shell=True)

  @classmethod
  def tearDownClass(cls):
    subprocess.check_call('gsutil -m rm -r %s' % cls._bucket_root, shell=True)

  def test_numerics(self):
    test_folder = os.path.join(self._bucket_root, 'test_numerics')
    input_file_path = os.path.join(test_folder, 'input.csv')
    output_folder = os.path.join(test_folder, 'test_output')
    file_io.recursive_create_dir(output_folder)

    file_io.write_string_to_file(
      input_file_path,
      '\n'.join(['%s,%s' % (i, 10 * i + 0.5) for i in range(100)]))

    analyze_data.run_cloud_analysis(
        output_dir=output_folder,
        csv_file_pattern=input_file_path,
        bigquery_table=None,
        schema=[{'name': 'col1', 'type': 'INTEGER'},
                {'name': 'col2', 'type': 'FLOAT'}],
        features={'col1': {'transform': 'scale'},
                  'col2': {'transform': 'identity'}})
    stats = json.loads(
        file_io.read_file_to_string(
            os.path.join(output_folder, analyze_data.constant.STATS_FILE)).decode())

    self.assertEqual(stats['num_examples'], 100)
    col = stats['column_stats']['col1']
    self.assertAlmostEqual(col['max'], 99.0)
    self.assertAlmostEqual(col['min'], 0.0)
    self.assertAlmostEqual(col['mean'], 49.5)

    col = stats['column_stats']['col2']
    self.assertAlmostEqual(col['max'], 990.5)
    self.assertAlmostEqual(col['min'], 0.5)
    self.assertAlmostEqual(col['mean'], 495.5)

  def test_categorical(self):
    test_folder = os.path.join(self._bucket_root, 'test_categorical')
    input_file_path = os.path.join(test_folder, 'input.csv')
    output_folder = os.path.join(test_folder, 'test_output')
    file_io.recursive_create_dir(output_folder)

    csv_file = ['red,car', 'red,truck', 'red,van', 'blue,bike', 'blue,train',
                'green,airplane']
    file_io.write_string_to_file(
      input_file_path,
      '\n'.join(csv_file))

    analyze_data.run_cloud_analysis(
        output_dir=output_folder,
        csv_file_pattern=input_file_path,
        bigquery_table=None,
        schema=[{'name': 'color', 'type': 'STRING'},
                {'name': 'transport', 'type': 'STRING'}],
        features={'color': {'transform': 'one_hot'},
                  'transport': {'transform': 'embedding'}})

    stats = json.loads(
        file_io.read_file_to_string(
            os.path.join(output_folder, analyze_data.constant.STATS_FILE)).decode())
    self.assertEqual(stats['column_stats']['color']['vocab_size'], 3)
    self.assertEqual(stats['column_stats']['transport']['vocab_size'], 6)

    # Color column.
    vocab_str = file_io.read_file_to_string(
      os.path.join(output_folder, analyze_data.constant.VOCAB_ANALYSIS_FILE % 'color'))
    vocab = pd.read_csv(six.StringIO(vocab_str),
                        header=None,
                        names=['color', 'count'])
    expected_vocab = pd.DataFrame(
        {'color': ['red', 'blue', 'green'], 'count': [3, 2, 1]},
        columns=['color', 'count'])
    pd.util.testing.assert_frame_equal(vocab, expected_vocab)

    # transport column.
    vocab_str = file_io.read_file_to_string(
        os.path.join(output_folder,
                     analyze_data.constant.VOCAB_ANALYSIS_FILE % 'transport'))
    vocab = pd.read_csv(six.StringIO(vocab_str),
                        header=None,
                        names=['transport', 'count'])
    self.assertEqual(vocab['count'].tolist(), [1 for i in range(6)])
    self.assertEqual(vocab['transport'].tolist(),
                     ['airplane', 'bike', 'car', 'train', 'truck', 'van'])

  def test_text(self):
    test_folder = os.path.join(self._bucket_root, 'test_text')
    input_file_path = os.path.join(test_folder, 'input.csv')
    output_folder = os.path.join(test_folder, 'test_output')
    file_io.recursive_create_dir(output_folder)

    csv_file = ['the quick brown fox,raining in kir',
                'quick   brown brown chicken,raining in pdx']
    file_io.write_string_to_file(
      input_file_path,
      '\n'.join(csv_file))

    analyze_data.run_cloud_analysis(
        output_dir=output_folder,
        csv_file_pattern=input_file_path,
        bigquery_table=None,
        schema=[{'name': 'col1', 'type': 'STRING'},
                {'name': 'col2', 'type': 'STRING'}],
        features={'col1': {'transform': 'bag_of_words'},
                  'col2': {'transform': 'tfidf'}})

    stats = json.loads(
        file_io.read_file_to_string(
            os.path.join(output_folder, analyze_data.constant.STATS_FILE)).decode())
    self.assertEqual(stats['column_stats']['col1']['vocab_size'], 5)
    self.assertEqual(stats['column_stats']['col2']['vocab_size'], 4)

    vocab_str = file_io.read_file_to_string(
        os.path.join(output_folder,
                     analyze_data.constant.VOCAB_ANALYSIS_FILE % 'col1'))
    vocab = pd.read_csv(six.StringIO(vocab_str),
                        header=None,
                        names=['col1', 'count'])
    self.assertEqual(vocab['col1'].tolist(),
                     ['brown', 'quick', 'chicken', 'fox', 'the', ])
    self.assertEqual(vocab['count'].tolist(), [2, 2, 1, 1, 1])

    vocab_str = file_io.read_file_to_string(
        os.path.join(output_folder,
                     analyze_data.constant.VOCAB_ANALYSIS_FILE % 'col2'))
    vocab = pd.read_csv(six.StringIO(vocab_str),
                        header=None,
                        names=['col2', 'count'])
    self.assertEqual(vocab['col2'].tolist(), ['in', 'raining', 'kir', 'pdx'])
    self.assertEqual(vocab['count'].tolist(), [2, 2, 1, 1])


if __name__ == '__main__':
    unittest.main()
