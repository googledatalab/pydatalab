from __future__ import absolute_import
from __future__ import print_function

import json
import os
import shutil
import sys
import tempfile
import unittest
import pandas as pd
import six

from tensorflow.python.lib.io import file_io


# To make 'import analyze' work without installing it.
CODE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '', 'xgboost'))
sys.path.append(CODE_PATH)


from trainer import feature_analysis as feature_analysis  # noqa: E303
import analyze  # noqa: E303


class TestConfigFiles(unittest.TestCase):
  """Tests for checking the format between the schema and features files."""

  def test_expand_defaults_do_nothing(self):
    schema = [{'name': 'col1', 'type': 'FLOAT'},
              {'name': 'col2', 'type': 'INTEGER'}]
    features = {'col1': {'transform': 'x'},
                'col2': {'transform': 'y'}}
    expected_features = {
        'col1': {'transform': 'x', 'source_column': 'col1'},
        'col2': {'transform': 'y', 'source_column': 'col2'}}

    feature_analysis.expand_defaults(schema, features)

    # Nothing should change.
    self.assertEqual(expected_features, features)

  def test_expand_defaults_unknown_schema_type(self):
    schema = [{'name': 'col1', 'type': 'BYTES'},
              {'name': 'col2', 'type': 'INTEGER'}]
    features = {'col1': {'transform': 'x'},
                'col2': {'transform': 'y'}}

    with self.assertRaises(ValueError):
      feature_analysis.expand_defaults(schema, features)

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

    feature_analysis.expand_defaults(schema, features)

    self.assertEqual(
      features,
      {'col1': {'transform': 'x', 'source_column': 'col1'},
       'col2': {'transform': 'y', 'source_column': 'col2'},
       'col3': {'transform': 'z', 'source_column': 'col3'},
       'col4': {'transform': 'identity', 'source_column': 'col4'},
       'col5': {'transform': 'identity', 'source_column': 'col5'},
       'col6': {'transform': 'one_hot', 'source_column': 'col6'}})


class TestLocalAnalyze(unittest.TestCase):
  """Test local analyze functions."""

  def test_numerics(self):
    output_folder = tempfile.mkdtemp()
    input_file_path = tempfile.mkstemp(dir=output_folder)[1]
    try:
      file_io.write_string_to_file(
        input_file_path,
        '\n'.join(['%s,%s,%s' % (i, 10 * i + 0.5, i + 0.5) for i in range(100)]))

      schema = [{'name': 'col1', 'type': 'INTEGER'},
                {'name': 'col2', 'type': 'FLOAT'},
                {'name': 'col3', 'type': 'FLOAT'}]
      features = {'col1': {'transform': 'scale', 'source_column': 'col1'},
                  'col2': {'transform': 'identity', 'source_column': 'col2'},
                  'col3': {'transform': 'target'}}
      feature_analysis.run_local_analysis(
          output_folder, [input_file_path], schema, features)

      stats = json.loads(
          file_io.read_file_to_string(
              os.path.join(output_folder, analyze.constant.STATS_FILE)).decode())

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
      csv_file = ['red,apple', 'red,pepper', 'red,apple', 'blue,grape',
                  'blue,apple', 'green,pepper']
      file_io.write_string_to_file(
        input_file_path,
        '\n'.join(csv_file))

      schema = [{'name': 'color', 'type': 'STRING'},
                {'name': 'type', 'type': 'STRING'}]
      features = {'color': {'transform': 'one_hot', 'source_column': 'color'},
                  'type': {'transform': 'target'}}
      feature_analysis.run_local_analysis(
        output_folder, [input_file_path], schema, features)

      stats = json.loads(
          file_io.read_file_to_string(
              os.path.join(output_folder, analyze.constant.STATS_FILE)).decode())
      self.assertEqual(stats['column_stats']['color']['vocab_size'], 3)

      # Color column.
      vocab_str = file_io.read_file_to_string(
        os.path.join(output_folder, analyze.constant.VOCAB_ANALYSIS_FILE % 'color'))
      vocab = pd.read_csv(six.StringIO(vocab_str),
                          header=None,
                          names=['color', 'count'])
      expected_vocab = pd.DataFrame(
          {'color': ['red', 'blue', 'green'], 'count': [3, 2, 1]},
          columns=['color', 'count'])
      pd.util.testing.assert_frame_equal(vocab, expected_vocab)

    finally:
      shutil.rmtree(output_folder)

  def test_text(self):
    output_folder = tempfile.mkdtemp()
    input_file_path = tempfile.mkstemp(dir=output_folder)[1]
    try:
      csv_file = ['the quick brown fox,cat1|cat2,true',
                  'quick   brown brown chicken,cat2|cat3|cat4,false']
      file_io.write_string_to_file(
        input_file_path,
        '\n'.join(csv_file))

      schema = [{'name': 'col1', 'type': 'STRING'},
                {'name': 'col2', 'type': 'STRING'},
                {'name': 'col3', 'type': 'STRING'}]
      features = {'col1': {'transform': 'multi_hot', 'source_column': 'col1'},
                  'col2': {'transform': 'multi_hot', 'source_column': 'col2', 'separator': '|'},
                  'col3': {'transform': 'target'}}
      feature_analysis.run_local_analysis(
        output_folder, [input_file_path], schema, features)

      stats = json.loads(
          file_io.read_file_to_string(
              os.path.join(output_folder, analyze.constant.STATS_FILE)).decode())
      self.assertEqual(stats['column_stats']['col1']['vocab_size'], 5)
      self.assertEqual(stats['column_stats']['col2']['vocab_size'], 4)

      vocab_str = file_io.read_file_to_string(
          os.path.join(output_folder,
                       analyze.constant.VOCAB_ANALYSIS_FILE % 'col1'))
      vocab = pd.read_csv(six.StringIO(vocab_str),
                          header=None,
                          names=['col1', 'count'])

      # vocabs are sorted by count only
      col1_vocab = vocab['col1'].tolist()
      self.assertItemsEqual(col1_vocab[:2], ['brown', 'quick'])
      self.assertItemsEqual(col1_vocab[2:], ['chicken', 'fox', 'the'])
      self.assertEqual(vocab['count'].tolist(), [2, 2, 1, 1, 1])

      vocab_str = file_io.read_file_to_string(
          os.path.join(output_folder,
                       analyze.constant.VOCAB_ANALYSIS_FILE % 'col2'))
      vocab = pd.read_csv(six.StringIO(vocab_str),
                          header=None,
                          names=['col2', 'count'])

      # vocabs are sorted by count only
      col2_vocab = vocab['col2'].tolist()
      self.assertItemsEqual(col2_vocab, ['cat2', 'cat1', 'cat3', 'cat4'])
      self.assertEqual(vocab['count'].tolist(), [2, 1, 1, 1])
    finally:
      shutil.rmtree(output_folder)


if __name__ == '__main__':
    unittest.main()
