from __future__ import absolute_import
from __future__ import print_function

import base64
import cStringIO
from PIL import Image
import json
import math
import numpy as np
import os
import shutil
import sys
import tempfile
import unittest
import tensorflow as tf

from tensorflow.python.lib.io import file_io

# To make 'import analyze' work without installing it.
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'mltoolbox', 'code_free_ml', 'trainer')))

import feature_transforms  # noqa: E303


class TestGraphBuilding(unittest.TestCase):
  """Test the TITO functions work and can produce a working TF graph."""

  def _run_graph(self, analysis_path, features, schema, stats, predict_data):
    """Runs the preprocessing graph.

    Args:
      analysis_path: path to folder containing analysis output. Should contain
          the stats file.
      features: features dict
      schema: schema list
      stats: stats dict
      predict_data: list of csv strings
    """
    stats = {'column_stats': {}}
    with tf.Graph().as_default():
      with tf.Session().as_default() as session:
        outputs, labels, inputs = feature_transforms.build_csv_serving_tensors(
            analysis_path, features, schema, stats, keep_target=False)
        feed_inputs = {inputs['csv_example']: predict_data}

        session.run(tf.tables_initializer())
        result = session.run(outputs, feed_dict=feed_inputs)
        return result

  def test_make_transform_graph_numerics(self):
    output_folder = tempfile.mkdtemp()
    stats_file_path = os.path.join(output_folder, feature_transforms.STATS_FILE)
    try:
      stats = {'column_stats':
                {'num1': {'max': 10.0, 'mean': 9.5, 'min': 0.0},  # noqa
                 'num2': {'max': 1.0, 'mean': 2.0, 'min': -1.0},
                 'num3': {'max': 10.0, 'mean': 2.0, 'min': 5.0}}}
      schema = [{'name': 'num1', 'type': 'FLOAT'},
                {'name': 'num2', 'type': 'FLOAT'},
                {'name': 'num3', 'type': 'INTEGER'}]
      features = {'num1': {'transform': 'identity', 'source_column': 'num1'},
                  'num2': {'transform': 'scale', 'value': 10, 'source_column': 'num2'},
                  'num3': {'transform': 'scale', 'source_column': 'num3'}}
      input_data = ['5.0,-1.0,10',
                    '10.0,1.0,5',
                    '15.0,0.5,7']
      file_io.write_string_to_file(
          stats_file_path,
          json.dumps(stats))

      results = self._run_graph(output_folder, features, schema, stats, input_data)

      for result, expected_result in zip(results['num1'].flatten().tolist(),
                                         [5, 10, 15]):
        self.assertAlmostEqual(result, expected_result)

      for result, expected_result in zip(results['num2'].flatten().tolist(),
                                         [-10, 10, 5]):
        self.assertAlmostEqual(result, expected_result)

      for result, expected_result in zip(results['num3'].flatten().tolist(),
                                         [1, -1, (7.0 - 5) * 2.0 / 5.0 - 1]):
        self.assertAlmostEqual(result, expected_result)
    finally:
      shutil.rmtree(output_folder)

  def test_make_transform_graph_category(self):
    output_folder = tempfile.mkdtemp()
    try:
      file_io.write_string_to_file(
          os.path.join(output_folder, feature_transforms.VOCAB_ANALYSIS_FILE % 'cat1'),
          '\n'.join(['red,300', 'blue,200', 'green,100']))

      file_io.write_string_to_file(
          os.path.join(output_folder, feature_transforms.VOCAB_ANALYSIS_FILE % 'cat2'),
          '\n'.join(['pizza,300', 'ice_cream,200', 'cookies,100']))

      stats = {'column_stats': {}}  # stats file needed but unused.
      file_io.write_string_to_file(
          os.path.join(output_folder, feature_transforms.STATS_FILE),
          json.dumps(stats))

      schema = [{'name': 'cat1', 'type': 'STRING'}, {'name': 'cat2', 'type': 'STRING'}]
      features = {'cat1': {'transform': 'one_hot', 'source_column': 'cat1'},
                  'cat2': {'transform': 'embedding', 'source_column': 'cat2'}}
      input_data = ['red,pizza',
                    'blue,',
                    'green,extra']

      results = self._run_graph(output_folder, features, schema, stats, input_data)

      for result, expected_result in zip(results['cat1'].flatten().tolist(), [0, 1, 2]):
        self.assertEqual(result, expected_result)

      for result, expected_result in zip(results['cat2'].flatten().tolist(),
                                         [0, 3, 3]):
        self.assertEqual(result, expected_result)
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
          os.path.join(output_folder, feature_transforms.VOCAB_ANALYSIS_FILE % 'cat1'),
          '\n'.join(['red,2', 'blue,2', 'green,1']))

      stats = {'column_stats': {}, 'num_examples': 4}
      file_io.write_string_to_file(
          os.path.join(output_folder, feature_transforms.STATS_FILE),
          json.dumps(stats))

      # decode_csv does not like 1 column files with an empty row, so add
      # a key column
      schema = [{'name': 'key', 'type': 'STRING'},
                {'name': 'cat1', 'type': 'STRING'}]
      features = {'key': {'transform': 'key', 'source_column': 'key'},
                  'cat1': {'transform': 'tfidf', 'source_column': 'cat1'}}
      input_data = ['0,red red red',    # doc 0
                    '1,red green red',  # doc 1
                    '2,blue',           # doc 2
                    '3,blue blue',      # doc 3
                    '4,',               # doc 4
                    '5,brown',          # doc 5
                    '6,brown blue']     # doc 6

      results = self._run_graph(output_folder, features, schema, stats, input_data)

      # indices are in the form [doc id, vocab id]
      expected_indices = [[0, 0], [0, 1], [0, 2],
                          [1, 0], [1, 1], [1, 2],
                          [2, 0],
                          [3, 0], [3, 1],
                          [5, 0],
                          [6, 0], [6, 1]]
      expected_ids = [0, 0, 0, 0, 2, 0, 1, 1, 1, 3, 3, 1]
      self.assertEqual(results['cat1_ids'].indices.tolist(), expected_indices)
      self.assertEqual(results['cat1_ids'].dense_shape.tolist(), [7, 3])
      self.assertEqual(results['cat1_ids'].values.tolist(), expected_ids)

      # Note, these are natural logs.
      log_4_3 = math.log(4.0 / 3.0)
      expected_weights = [
          1.0 / 3.0 * log_4_3, 1.0 / 3.0 * log_4_3, 1.0 / 3.0 * log_4_3,  # doc 0
          1.0 / 3.0 * log_4_3, 1.0 / 3.0 * math.log(2.0), 1.0 / 3.0 * log_4_3,  # doc 1
          math.log(4.0 / 3.0),  # doc 2
          1.0 / 2.0 * log_4_3, 1.0 / 2.0 * log_4_3,  # doc 3
          math.log(4.0),  # doc 5
          1.0 / 2.0 * math.log(4.0), 1.0 / 2.0 * log_4_3]  # doc 6

      self.assertEqual(results['cat1_weights'].indices.tolist(), expected_indices)
      self.assertEqual(results['cat1_weights'].dense_shape.tolist(), [7, 3])
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
                       feature_transforms.VOCAB_ANALYSIS_FILE % 'cat1'),
          '\n'.join(['red,2', 'blue,2', 'green,1']))

      stats = {'column_stats': {}}
      file_io.write_string_to_file(
          os.path.join(output_folder, feature_transforms.STATS_FILE),
          json.dumps(stats))  # Stats file needed but unused.

      # decode_csv does not like 1 column files with an empty row, so add
      # a key column
      schema = [{'name': 'key', 'type': 'STRING'},
                {'name': 'cat1', 'type': 'STRING'}]
      features = {'key': {'transform': 'key', 'source_column': 'key'},
                  'cat1': {'transform': 'bag_of_words', 'source_column': 'cat1'}}
      input_data = ['0,red red red',    # doc 0
                    '1,red green red',  # doc 1
                    '2,blue',           # doc 2
                    '3,blue blue',      # doc 3
                    '4,',               # doc 4
                    '5,brown',          # doc 5
                    '6,brown blue']     # doc 6

      results = self._run_graph(output_folder, features, schema, stats, input_data)

      # indices are in the form [doc id, vocab id]
      expected_indices = [[0, 0], [0, 1], [0, 2],
                          [1, 0], [1, 1], [1, 2],
                          [2, 0],
                          [3, 0], [3, 1],
                          [5, 0],
                          [6, 0], [6, 1]]

      # Note in doc 6, is is blue, then brown.
      # doc id            0  0  0  1  1  1  2  3  3  5  6  6
      expected_ids =     [0, 0, 0, 0, 2, 0, 1, 1, 1, 3, 3, 1] # noqa
      expected_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
      self.assertEqual(results['cat1_ids'].indices.tolist(), expected_indices)
      self.assertEqual(results['cat1_ids'].dense_shape.tolist(), [7, 3])
      self.assertEqual(results['cat1_ids'].values.tolist(), expected_ids)

      self.assertEqual(results['cat1_weights'].indices.tolist(),
                       expected_indices)
      self.assertEqual(results['cat1_weights'].dense_shape.tolist(), [7, 3])
      self.assertEqual(results['cat1_weights'].values.size,
                       len(expected_weights))
      for weight, exp_weight in zip(results['cat1_weights'].values.tolist(),
                                    expected_weights):
        self.assertAlmostEqual(weight, exp_weight)

    finally:
      shutil.rmtree(output_folder)

  def test_make_transform_graph_images(self):

    print('Testing make_transform_graph with image_to_vec.' +
          'It may take a few minutes because it needs to download a large inception checkpoint.')

    def _open_and_encode_image(img_url):
      with file_io.FileIO(img_url, 'r') as f:
        img = Image.open(f).convert('RGB')
        output = cStringIO.StringIO()
        img.save(output, 'jpeg')
      return base64.urlsafe_b64encode(output.getvalue())

    try:
      output_folder = tempfile.mkdtemp()
      stats_file_path = os.path.join(output_folder, feature_transforms.STATS_FILE)
      stats = {'column_stats': {}}
      file_io.write_string_to_file(stats_file_path, json.dumps(stats))

      schema = [{'name': 'img', 'type': 'STRING'}]
      features = {'img': {'transform': 'image_to_vec', 'source_column': 'img'}}

      img_string1 = _open_and_encode_image(
          'gs://cloud-ml-data/img/flower_photos/daisy/15207766_fc2f1d692c_n.jpg')
      img_string2 = _open_and_encode_image(
          'gs://cloud-ml-data/img/flower_photos/dandelion/8980164828_04fbf64f79_n.jpg')
      input_data = [img_string1, img_string2]
      results = self._run_graph(output_folder, features, schema, stats, input_data)
      embeddings = results['img']
      self.assertEqual(len(embeddings), 2)
      self.assertEqual(len(embeddings[0]), 2048)
      self.assertEqual(embeddings[0].dtype, np.float32)
      self.assertTrue(any(x != 0.0 for x in embeddings[1]))

    finally:
      shutil.rmtree(output_folder)


if __name__ == '__main__':
  unittest.main()
