from __future__ import absolute_import
from __future__ import print_function

import copy
import json
import math
import os
import shutil
import subprocess
import tempfile
import uuid
import unittest
import pandas as pd
import six
import tensorflow as tf

from tensorflow.python.lib.io import file_io
from tensorflow_transform import impl_helper

import google.datalab as dl
import google.datalab.bigquery as bq

import analyze_data


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
              os.path.join(output_folder, analyze_data.STATS_FILE)).decode())

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
              os.path.join(output_folder, analyze_data.STATS_FILE)).decode())
      self.assertEqual(stats['column_stats']['color']['vocab_size'], 3)
      self.assertEqual(stats['column_stats']['transport']['vocab_size'], 6)

      # Color column.
      vocab_str = file_io.read_file_to_string(
        os.path.join(output_folder, analyze_data.VOCAB_ANALYSIS_FILE % 'color'))
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
                       analyze_data.VOCAB_ANALYSIS_FILE % 'transport'))
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
              os.path.join(output_folder, analyze_data.STATS_FILE)).decode())
      self.assertEqual(stats['column_stats']['col1']['vocab_size'], 5)
      self.assertEqual(stats['column_stats']['col2']['vocab_size'], 4)

      vocab_str = file_io.read_file_to_string(
          os.path.join(output_folder,
                       analyze_data.VOCAB_ANALYSIS_FILE % 'col1'))
      vocab = pd.read_csv(six.StringIO(vocab_str),
                          header=None,
                          names=['col1', 'count'])
      self.assertEqual(vocab['col1'].tolist(),
                       ['quick', 'brown', 'the', 'fox', 'chicken'])
      self.assertEqual(vocab['count'].tolist(), [2, 2, 1, 1, 1])

      vocab_str = file_io.read_file_to_string(
          os.path.join(output_folder,
                       analyze_data.VOCAB_ANALYSIS_FILE % 'col2'))
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
              os.path.join(output_folder, analyze_data.STATS_FILE)).decode())

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
            os.path.join(output_folder, analyze_data.STATS_FILE)).decode())

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
            os.path.join(output_folder, analyze_data.STATS_FILE)).decode())
    self.assertEqual(stats['column_stats']['color']['vocab_size'], 3)
    self.assertEqual(stats['column_stats']['transport']['vocab_size'], 6)

    # Color column.
    vocab_str = file_io.read_file_to_string(
      os.path.join(output_folder, analyze_data.VOCAB_ANALYSIS_FILE % 'color'))
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
                     analyze_data.VOCAB_ANALYSIS_FILE % 'transport'))
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
            os.path.join(output_folder, analyze_data.STATS_FILE)).decode())
    self.assertEqual(stats['column_stats']['col1']['vocab_size'], 5)
    self.assertEqual(stats['column_stats']['col2']['vocab_size'], 4)

    vocab_str = file_io.read_file_to_string(
        os.path.join(output_folder,
                     analyze_data.VOCAB_ANALYSIS_FILE % 'col1'))
    vocab = pd.read_csv(six.StringIO(vocab_str),
                        header=None,
                        names=['col1', 'count'])
    self.assertEqual(vocab['col1'].tolist(),
                     ['brown', 'quick', 'chicken', 'fox', 'the', ])
    self.assertEqual(vocab['count'].tolist(), [2, 2, 1, 1, 1])

    vocab_str = file_io.read_file_to_string(
        os.path.join(output_folder,
                     analyze_data.VOCAB_ANALYSIS_FILE % 'col2'))
    vocab = pd.read_csv(six.StringIO(vocab_str),
                        header=None,
                        names=['col2', 'count'])
    self.assertEqual(vocab['col2'].tolist(), ['in', 'raining', 'kir', 'pdx'])
    self.assertEqual(vocab['count'].tolist(), [2, 2, 1, 1])


class TestGraphBuilding(unittest.TestCase):
  """Test the TITO functions work and can produce a working TF graph."""

  def _run_graph(self, model_path, predict_data):
    """Runs the preprocessing graph.

    Args:
      model_path: path to a folder that contains the save_model.pb file_io
      predict_data: batched feed dict. Eample {'column_name': [1,2,3]}
    """
    # Cannot call
    # bundle_shim.load_session_bundle_or_saved_model_bundle_from_path directly
    # as tft changes the in/output tensor names.

    g = tf.Graph()
    session = tf.Session(graph=g)
    with g.as_default():
      inputs, outputs = impl_helper.load_transform_fn_def(model_path)
      session.run(tf.tables_initializer())
      feed = {inputs[key]: value for key, value in six.iteritems(predict_data)}
      result = session.run(outputs, feed_dict=feed)

    return result

  def test_make_transform_graph_numerics(self):
    output_folder = tempfile.mkdtemp()
    stats_file_path = os.path.join(output_folder, analyze_data.STATS_FILE)
    try:
      file_io.write_string_to_file(
          stats_file_path,
          json.dumps({'column_stats':
                        {'num1': {'max': 10.0, 'mean': 9.5, 'min': 0.0},  # noqa
                         'num2': {'max': 1.0, 'mean': 2.0, 'min': -1.0},
                         'num3': {'max': 10.0, 'mean': 2.0, 'min': 5.0}}}))
      analyze_data.make_transform_graph(
        output_folder,
        [{'name': 'num1', 'type': 'FLOAT'},
         {'name': 'num2', 'type': 'FLOAT'},
         {'name': 'num3', 'type': 'INTEGER'}],
        {'num1': {'transform': 'identity'},
         'num2': {'transform': 'scale', 'value': 10},
         'num3': {'transform': 'scale'}})

      model_path = os.path.join(output_folder, 'transform_fn')
      self.assertTrue(os.path.isfile(os.path.join(model_path, 'saved_model.pb')))

      results = self._run_graph(model_path, {'num1': [5, 10, 15],
                                             'num2': [-1, 1, 0.5],
                                             'num3': [10, 5, 7]})

      for result, expected_result in zip(results['num1'].tolist(), [5, 10, 15]):
        self.assertAlmostEqual(result, expected_result)

      for result, expected_result in zip(results['num2'].tolist(),
                                         [-10, 10, 5]):
        self.assertAlmostEqual(result, expected_result)

      for result, expected_result in zip(results['num3'].tolist(),
                                         [1, -1, (7.0 - 5) * 2.0 / 5.0 - 1]):
        self.assertAlmostEqual(result, expected_result)
    finally:
      shutil.rmtree(output_folder)

  def test_make_transform_graph_numerics_gcs(self):
    """Input and output of this test is on GCS."""

    output_folder = 'gs://temp_pydatalab_test_%s' % uuid.uuid4().hex
    subprocess.check_call('gsutil mb %s' % output_folder, shell=True)
    stats_file_path = os.path.join(output_folder, analyze_data.STATS_FILE)
    try:
      file_io.write_string_to_file(
          stats_file_path,
          json.dumps({'column_stats':
                        {'num1': {'max': 10.0, 'mean': 9.5, 'min': 0.0},  # noqa
                         'num2': {'max': 1.0, 'mean': 2.0, 'min': -1.0},
                         'num3': {'max': 10.0, 'mean': 2.0, 'min': 5.0}}}))
      analyze_data.make_transform_graph(
        output_folder,
        [{'name': 'num1', 'type': 'FLOAT'},
         {'name': 'num2', 'type': 'FLOAT'},
         {'name': 'num3', 'type': 'INTEGER'}],
        {'num1': {'transform': 'identity'},
         'num2': {'transform': 'scale', 'value': 10},
         'num3': {'transform': 'scale'}})

      model_path = os.path.join(output_folder, 'transform_fn')
      self.assertTrue(file_io.file_exists(os.path.join(model_path, 'saved_model.pb')))

      results = self._run_graph(model_path, {'num1': [5, 10, 15],
                                             'num2': [-1, 1, 0.5],
                                             'num3': [10, 5, 7]})

      for result, expected_result in zip(results['num1'].tolist(), [5, 10, 15]):
        self.assertAlmostEqual(result, expected_result)

      for result, expected_result in zip(results['num2'].tolist(),
                                         [-10, 10, 5]):
        self.assertAlmostEqual(result, expected_result)

      for result, expected_result in zip(results['num3'].tolist(),
                                         [1, -1, (7.0 - 5) * 2.0 / 5.0 - 1]):
        self.assertAlmostEqual(result, expected_result)
    finally:
      subprocess.check_call('gsutil -m rm -r %s' % output_folder, shell=True)

  def test_make_transform_graph_category(self):
    output_folder = tempfile.mkdtemp()
    try:
      file_io.write_string_to_file(
          os.path.join(output_folder, analyze_data.VOCAB_ANALYSIS_FILE % 'cat1'),
          '\n'.join(['red,300', 'blue,200', 'green,100']))

      file_io.write_string_to_file(
          os.path.join(output_folder, analyze_data.VOCAB_ANALYSIS_FILE % 'cat2'),
          '\n'.join(['pizza,300', 'ice_cream,200', 'cookies,100']))

      file_io.write_string_to_file(
          os.path.join(output_folder, analyze_data.STATS_FILE),
          json.dumps({}))  # stats file needed but unused.

      analyze_data.make_transform_graph(
        output_folder,
        [{'name': 'cat1', 'type': 'STRING'}, {'name': 'cat2', 'type': 'STRING'}],
        {'cat1': {'transform': 'one_hot'},
         'cat2': {'transform': 'embedding'}})

      model_path = os.path.join(output_folder, 'transform_fn')
      self.assertTrue(os.path.isfile(os.path.join(model_path, 'saved_model.pb')))

      results = self._run_graph(model_path, {'cat1': ['red', 'blue', 'green'],
                                             'cat2': ['pizza', '', 'extra']})

      for result, expected_result in zip(results['cat1'].tolist(), [0, 1, 2]):
        self.assertAlmostEqual(result, expected_result)

      for result, expected_result in zip(results['cat2'].tolist(),
                                         [0, 3, 3]):
        self.assertAlmostEqual(result, expected_result)
    finally:
      shutil.rmtree(output_folder)

  def test_make_transform_graph_text_tfidf(self):
    output_folder = tempfile.mkdtemp()
    try:
      # vocab  id
      # red    0
      # blue   1
      # green  2
      # oov    3 (out of vocab)
      # corpus size aka num_examples = 4
      # IDF: log(num_examples/(1+number of examples that have this token))
      #  red: log(4/3)
      #  blue: log(4/3)
      #  green: log(4/2)
      #  oov:  log(4/1)
      file_io.write_string_to_file(
          os.path.join(output_folder, analyze_data.VOCAB_ANALYSIS_FILE % 'cat1'),
          '\n'.join(['red,2', 'blue,2', 'green,1']))

      file_io.write_string_to_file(
          os.path.join(output_folder, analyze_data.STATS_FILE),
          json.dumps({'num_examples': 4}))

      analyze_data.make_transform_graph(
        output_folder,
        [{'name': 'cat1', 'type': 'STRING'}],
        {'cat1': {'transform': 'tfidf'}})

      model_path = os.path.join(output_folder, 'transform_fn')
      self.assertTrue(os.path.isfile(os.path.join(model_path, 'saved_model.pb')))

      results = self._run_graph(
          model_path,
          {'cat1': ['red red red',    # doc 0
                    'red green red',  # doc 1
                    'blue',           # doc 2
                    'blue blue',      # doc 3
                    '',               # doc 4
                    'brown',          # doc 5
                    'brown blue']})   # doc 6

      # indices are in the form [doc id, vocab id]
      expected_indices = [[0, 0],
                          [1, 0], [1, 1],
                          [2, 0],
                          [3, 0],
                          [5, 0],
                          [6, 0], [6, 1]]
      expected_ids = [0, 0, 2, 1, 1, 3, 1, 3]  # Note in doc 6, it is blue, then brown.
      self.assertEqual(results['cat1_ids'].indices.tolist(), expected_indices)
      self.assertEqual(results['cat1_ids'].dense_shape.tolist(), [7, 4])
      self.assertEqual(results['cat1_ids'].values.tolist(), expected_ids)

      # Note, these are natural logs.
      expected_weights = [
          math.log(4.0 / 3.0),  # doc 0
          2.0 / 3.0 * math.log(4.0 / 3.0), 1.0 / 3.0 * math.log(2.0),  # doc 1
          math.log(4.0 / 3.0),  # doc 2
          math.log(4.0 / 3.0),  # doc 3
          math.log(4.0),  # doc 5
          1.0 / 2.0 * math.log(4.0 / 3.0), 1.0 / 2.0 * math.log(4.0)]  # doc 6

      self.assertEqual(results['cat1_weights'].indices.tolist(), expected_indices)
      self.assertEqual(results['cat1_weights'].dense_shape.tolist(), [7, 4])
      self.assertEqual(results['cat1_weights'].values.size, len(expected_weights))
      for weight, expected_weight in zip(results['cat1_weights'].values.tolist(), expected_weights):
        self.assertAlmostEqual(weight, expected_weight)

    finally:
      shutil.rmtree(output_folder)

  def test_make_transform_graph_text_bag_of_words(self):
    output_folder = tempfile.mkdtemp()
    try:
      # vocab  id
      # red    0
      # blue   1
      # green  2
      # oov    3 (out of vocab)
      file_io.write_string_to_file(
          os.path.join(output_folder,
                       analyze_data.VOCAB_ANALYSIS_FILE % 'cat1'),
          '\n'.join(['red,2', 'blue,2', 'green,1']))

      file_io.write_string_to_file(
          os.path.join(output_folder, analyze_data.STATS_FILE),
          json.dumps({}))  # Stats file needed but unused.

      analyze_data.make_transform_graph(
        output_folder,
        [{'name': 'cat1', 'type': 'STRING'}],
        {'cat1': {'transform': 'bag_of_words'}})

      model_path = os.path.join(output_folder, 'transform_fn')
      self.assertTrue(os.path.isfile(os.path.join(model_path,
                                                  'saved_model.pb')))

      results = self._run_graph(
          model_path,
          {'cat1': ['red red red',    # doc 0
                    'red green red',  # doc 1
                    'blue',           # doc 2
                    'blue blue',      # doc 3
                    '',               # doc 4
                    'brown',          # doc 5
                    'brown blue']})   # doc 6

      # indices are in the form [doc id, vocab id]
      expected_indices = [[0, 0],
                          [1, 0], [1, 1],
                          [2, 0],
                          [3, 0],
                          [5, 0],
                          [6, 0], [6, 1]]

      # Note in doc 6, is is blue, then brown.
      # doc id            0  1  1  2  3  5  6  6
      expected_ids =     [0, 0, 2, 1, 1, 3, 1, 3]  # noqa
      expected_weights = [3, 2, 1, 1, 2, 1, 1, 1]
      self.assertEqual(results['cat1_ids'].indices.tolist(), expected_indices)
      self.assertEqual(results['cat1_ids'].dense_shape.tolist(), [7, 4])
      self.assertEqual(results['cat1_ids'].values.tolist(), expected_ids)

      self.assertEqual(results['cat1_weights'].indices.tolist(),
                       expected_indices)
      self.assertEqual(results['cat1_weights'].dense_shape.tolist(), [7, 4])
      self.assertEqual(results['cat1_weights'].values.size,
                       len(expected_weights))
      for weight, exp_weight in zip(results['cat1_weights'].values.tolist(),
                                    expected_weights):
        self.assertAlmostEqual(weight, exp_weight)

    finally:
      shutil.rmtree(output_folder)


if __name__ == '__main__':
    unittest.main()
