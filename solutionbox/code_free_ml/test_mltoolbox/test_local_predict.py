from __future__ import absolute_import
import glob
import json
import os
import random
import shutil
import subprocess
import tempfile
import unittest
import pandas as pd
import six
import tensorflow as tf
from tensorflow.python.lib.io import file_io


CODE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'mltoolbox', 'code_free_ml', 'data'))


def make_csv_prediction_data(filename, num_rows):
  random.seed(12321)

  def _drop_out(x):
    # Make 5% of the data missing
    if random.uniform(0, 1) < 0.05:
      return ''
    return x

  with open(filename, 'w') as f:
    for i in range(num_rows):
      rand_num = random.randint(0, 20)
      rand_float = random.uniform(0, 10)
      rand_str = str(random.randint(0, 50)) + '.xyz'
      csv_line = '{rand_num},{rand_float},{rand_str}\n'.format(
          rand_num=rand_num, rand_float=rand_float, rand_str=rand_str)
      f.write(csv_line)


def make_prediction_graph(model_location):
  """Make a serving graph.

  inputs: 1 csv string of "int,float,string"
  outputs: each csv column
  """

  with tf.Graph().as_default():
    with tf.Session().as_default() as session:
      csv_placeholder = tf.placeholder(tf.string, shape=(None,))
      record_defaults = [[0], [0.0], ['MISSING']]
      parsed_tensors = tf.decode_csv(csv_placeholder, record_defaults)

      inputs = {'csv_line': csv_placeholder}
      outputs = {'rand_int': tf.expand_dims(parsed_tensors[0], -1),
                 'rand_float': tf.expand_dims(parsed_tensors[1], -1),
                 'rand_string': tf.expand_dims(parsed_tensors[2], -1)}

      # Build a saved model
      signature_inputs = {
          key: tf.saved_model.utils.build_tensor_info(tensor)
          for key, tensor in six.iteritems(inputs)
      }
      signature_outputs = {
          key: tf.saved_model.utils.build_tensor_info(tensor)
          for key, tensor in six.iteritems(outputs)
      }

      signature_def = tf.saved_model.signature_def_utils.build_signature_def(
          inputs=signature_inputs,
          outputs=signature_outputs,
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
      builder = tf.saved_model.builder.SavedModelBuilder(model_location)
      builder.add_meta_graph_and_variables(
          sess=session,
          tags=[tf.saved_model.tag_constants.SERVING],
          signature_def_map={'serving_default': signature_def})
      builder.save()


class TestPrediction(unittest.TestCase):
  """Tests local prediction

  Builds a tensorflow model, and checks local prediction works.
  """
  def test_prediction_to_json(self):
    """Test saving predictions to json file."""
    output_dir = tempfile.mkdtemp()
    try:
      make_csv_prediction_data(os.path.join(output_dir, 'input1.csv'), 50)
      make_csv_prediction_data(os.path.join(output_dir, 'input2.csv'), 100)
      make_csv_prediction_data(os.path.join(output_dir, 'input3.csv'), 89)

      model_location = os.path.join(output_dir, 'model')
      make_prediction_graph(model_location)

      cmd = ['python %s' % os.path.join(CODE_PATH, 'local_predict.py'),
             '--predict-data=%s' % os.path.join(output_dir, 'input*.csv'),
             '--trained-model-dir=%s' % os.path.join(output_dir, 'model'),
             '--output-location=%s' % output_dir,
             '--output-format=json',
             '--batch-size=16',
             '--shard-files']  # Makes 1 output prediction file per input file.

      subprocess.check_call(' '.join(cmd), shell=True)

      # Test 3 output prediction files was made
      json_files = glob.glob(os.path.join(output_dir, 'predictions*.json'))
      self.assertEqual(3, len(json_files))

      # Read just the first row to make sure it is valid json
      with open(json_files[0], 'r') as f:
        row = f.readline()
        json.loads(row)

      # Test the schema file was correctly made
      schema = json.loads(file_io.read_file_to_string(os.path.join(output_dir, 'schema.json')))
      expected_schema = [{"type": "FLOAT", "name": "rand_float"},
                         {"type": "INTEGER", "name": "rand_int"},
                         {"type": "STRING", "name": "rand_string"}]
      self.assertEqual(schema, expected_schema)
    finally:
      shutil.rmtree(output_dir)

  def test_prediction_to_csv(self):
    """Test saving predictions to csv file."""
    output_dir = tempfile.mkdtemp()
    try:
      make_csv_prediction_data(os.path.join(output_dir, 'input1.csv'), 50)
      make_csv_prediction_data(os.path.join(output_dir, 'input2.csv'), 100)
      make_csv_prediction_data(os.path.join(output_dir, 'input3.csv'), 89)

      model_location = os.path.join(output_dir, 'model')
      make_prediction_graph(model_location)

      cmd = ['python %s' % os.path.join(CODE_PATH, 'local_predict.py'),
             '--predict-data=%s' % os.path.join(output_dir, 'input*.csv'),
             '--trained-model-dir=%s' % os.path.join(output_dir, 'model'),
             '--output-location=%s' % output_dir,
             '--output-format=csv',
             '--batch-size=16',
             '--no-shard-files']

      subprocess.check_call(' '.join(cmd), shell=True)

      # Test 1 output prediction file was made
      csv_files = glob.glob(os.path.join(output_dir, 'predictions*.csv'))
      self.assertEqual(1, len(csv_files))

      # Read just the first row to make sure it is valid json
      data = pd.read_csv(csv_files[0],
                         header=None,
                         names=['rand_float', 'rand_int', 'rand_string'])
      self.assertEqual(50 + 100 + 89, len(data))
      self.assertEqual(3, len(data.columns))

      # Test the schema file was correctly made
      schema = json.loads(file_io.read_file_to_string(os.path.join(output_dir, 'schema.json')))
      expected_schema = [{"type": "FLOAT", "name": "rand_float"},
                         {"type": "INTEGER", "name": "rand_int"},
                         {"type": "STRING", "name": "rand_string"}]
      self.assertEqual(schema, expected_schema)
    finally:
      shutil.rmtree(output_dir)


if __name__ == '__main__':
    unittest.main()
