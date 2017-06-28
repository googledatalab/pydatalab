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

"""Test the datalab prediction with MLToolbox models."""
from __future__ import absolute_import
from __future__ import print_function


import base64
import csv
from io import BytesIO
import json
import logging
import os
import pandas as pd
from PIL import Image
import shutil
import six
import sys
import tempfile
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
import unittest


from google.datalab.contrib.mltoolbox import _local_predict


class TestLocalPredictions(unittest.TestCase):
  """Tests for checking the format between the schema and features files."""

  def setUp(self):
    self._logger = logging.getLogger('TestStructuredDataLogger')
    self._logger.setLevel(logging.DEBUG)
    if not self._logger.handlers:
      self._logger.addHandler(logging.StreamHandler(stream=sys.stdout))

    self._test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self._test_dir)

  def _create_model(self, dir_name):
    """Create a simple model that takes 'key', 'num1', 'text1', 'img_url1' input."""

    def _decode_jpg(image):
      img_buf = BytesIO()
      Image.new('RGB', (16, 16)).save(img_buf, 'jpeg')
      default_image_string = base64.urlsafe_b64encode(img_buf.getvalue())
      image = tf.where(tf.equal(image, ''), default_image_string, image)
      image = tf.decode_base64(image)
      image = tf.image.decode_jpeg(image, channels=3)
      image = tf.reshape(image, [-1])
      image = tf.reduce_max(image)
      return image

    model_dir = tempfile.mkdtemp()
    with tf.Session(graph=tf.Graph()) as sess:
      record_defaults = [
          tf.constant([0], dtype=tf.int64),
          tf.constant([0.0], dtype=tf.float32),
          tf.constant([''], dtype=tf.string),
          tf.constant([''], dtype=tf.string),
      ]
      placeholder = tf.placeholder(dtype=tf.string, shape=(None,), name='csv_input_placeholder')
      key_tensor, num_tensor, text_tensor, img_tensor = tf.decode_csv(placeholder, record_defaults)
      text_tensor = tf.string_to_number(text_tensor, tf.float32)
      img_tensor = tf.map_fn(_decode_jpg, img_tensor, back_prop=False, dtype=tf.uint8)
      img_tensor = tf.cast(img_tensor, tf.float32)
      stacked = tf.stack([num_tensor, text_tensor, img_tensor])
      min_tensor = tf.reduce_min(stacked, axis=0)
      max_tensor = tf.reduce_max(stacked, axis=0)

      predict_input_tensor = tf.saved_model.utils.build_tensor_info(placeholder)
      predict_signature_inputs = {"input": predict_input_tensor}
      predict_output_tensor1 = tf.saved_model.utils.build_tensor_info(min_tensor)
      predict_output_tensor2 = tf.saved_model.utils.build_tensor_info(max_tensor)
      predict_key_tensor = tf.saved_model.utils.build_tensor_info(key_tensor)
      predict_signature_outputs = {
        'key': predict_key_tensor,
        'var1': predict_output_tensor1,
        'var2': predict_output_tensor2
      }
      predict_signature_def = (
          tf.saved_model.signature_def_utils.build_signature_def(
              predict_signature_inputs, predict_signature_outputs,
              tf.saved_model.signature_constants.PREDICT_METHOD_NAME
          )
      )
      signature_def_map = {
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: predict_signature_def
      }
      model_dir = os.path.join(self._test_dir, dir_name)
      builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map=signature_def_map)
      builder.save(False)

    return model_dir

  def _create_test_data(self, embedding_images=False, missing_values=False, csv_data=False):
    image_path1 = os.path.join(self._test_dir, 'img1.jpg')
    image_path2 = os.path.join(self._test_dir, 'img2.jpg')
    image_path3 = os.path.join(self._test_dir, 'img3.jpg')
    Image.new('RGBA', size=(128, 128), color=(155, 211, 64)).save(image_path1, "JPEG")
    Image.new('RGB', size=(64, 64), color=(111, 21, 86)).save(image_path2, "JPEG")
    Image.new('RGBA', size=(16, 16), color=(255, 21, 1)).save(image_path3, "JPEG")

    data = [
        {'key': 1, 'num1': 4.1, 'text1': '32', 'img_url1': image_path1},
        {'key': 2, 'num1': 5.2, 'text1': '18', 'img_url1': image_path2},
        {'key': 5, 'num1': -12.0, 'text1': '22', 'img_url1': image_path3},
    ]
    if embedding_images:
      for d in data:
        with open(d['img_url1'], 'rb') as f:
          d['img_url1'] = base64.urlsafe_b64encode(f.read()).decode('ascii')

    if missing_values:
      del data[0]['num1']
      del data[1]['img_url1']

    if csv_data:
      csv_lines = []
      for d in data:
        buf = six.StringIO()
        writer = csv.DictWriter(buf, fieldnames=['key', 'num1', 'text1', 'img_url1'])
        writer.writerow(d)
        csv_lines.append(buf.getvalue().rstrip())
      data = csv_lines

    return data

  def _validate_results(self, df, show_image):
    expected_columns = set(['key', 'num1', 'text1', 'img_url1', 'var1', 'var2'])
    if show_image:
      expected_columns.add('img_url1_image')
    self.assertEqual(expected_columns, set(df.columns))
    self.assertEqual([1, 2, 5], df['key'].tolist())
    self.assertEqual(3, len(df.index))

  def test_predict_one_row(self):
    self.assertEqual(1,2)

  def test_predict(self):
    """ Test prediction on a model which accepts CSV lines "int64,float32,text,image_url".
    """

    model_dir = self._create_model('model1')
    headers = ['key', 'num1', 'text1', 'img_url1']

    # Test data being list of dict and list of csvlines, with and without missing values,
    # show or not show images.
    for missing_values in [True, False]:
      for csv_data in [True, False]:
        for show_image in [True, False]:
          self._logger.debug('LocalPredict: ' +
                             'missing_values=%s, csv_data=%s, show_image=%s' %
                             (missing_values, csv_data, show_image))
          test_data = self._create_test_data(False, missing_values, csv_data)
          df = _local_predict.get_prediction_results(
              model_dir, test_data, headers, ['img_url1'], False, show_image)
          self._validate_results(df, show_image)

    # Test data being dataframes, with and without missing values, and embedded images.
    for missing_values in [True, False]:
      self._logger.debug('LocalPredict: ' +
                         'missing_values=%s, DataFrame' % missing_values)
      test_data = self._create_test_data(True, missing_values, csv_data=False)
      df_s = pd.DataFrame(test_data).fillna('')
      df = _local_predict.get_prediction_results(model_dir, df_s, headers, None, False, False)
      self._validate_results(df, False)

  def _validate_schema_file(self, output_dir):
    with open(os.path.join(output_dir, 'predict_results_schema.json'), 'r') as f:
      schema = json.loads(f.read())

    expected_schema = [
        {"type": "INTEGER", "name": "key"},
        {"type": "FLOAT", "name": "var1"},
        {"type": "FLOAT", "name": "var2"}
    ]
    self.assertEqual(expected_schema, schema)

  def test_batch_predict(self):
    """ Test batch prediction on a model which accepts CSV lines "int64,float32,text,image_url".
    """

    self._logger.debug('Starting Local Batch Predict')
    model_dir = self._create_model('model2')
    test_data = self._create_test_data(embedding_images=True, missing_values=True, csv_data=True)
    prediction_source = os.path.join(self._test_dir, 'prediction.csv')
    output_dir = os.path.join(self._test_dir, 'prediction_output')
    with open(prediction_source, 'w') as f:
      f.write('\n'.join(test_data))

    # Test prediction output as csv file.
    _local_predict.local_batch_predict(model_dir, prediction_source,
                                       output_dir, 'csv', batch_size=2)
    self._validate_schema_file(output_dir)

    prediction_results_file = os.path.join(output_dir, 'predict_results_prediction.csv')
    df = pd.read_csv(prediction_results_file, header=None, names=['key', 'var1', 'var2'])
    self.assertEqual(3, len(df.index))
    self.assertEqual([1, 2, 5], list(df['key']))

    # Test prediction output as json file.
    _local_predict.local_batch_predict(model_dir, prediction_source,
                                       output_dir, 'json', batch_size=1)
    self._validate_schema_file(output_dir)

    prediction_results_file = os.path.join(output_dir, 'predict_results_prediction.json')
    results = []
    with open(prediction_results_file, 'r') as f:
      for l in f:
        results.append(json.loads(l))

    self.assertEqual(3, len(results))
    self.assertEqual([1, 2, 5], [x['key'] for x in results])


if __name__ == '__main__':
    unittest.main()
