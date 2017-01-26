# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import glob
import json
import os
import shutil
import subprocess
import tempfile
import unittest

import tensorflow as tf

import google.cloud.ml as ml

import e2e_functions


class TestPreprocess(unittest.TestCase):

  def setUp(self):
    self._test_dir = tempfile.mkdtemp()

    self._csv_filename = os.path.join(self._test_dir, 'raw_csv_data.csv')
    self._schema_filename = os.path.join(self._test_dir, 'schema.json')

  def tearDown(self):
    print('TestPreprocess: removing test dir: ' + self._test_dir)
    shutil.rmtree(self._test_dir)

  def testRegression(self):
    (schema, _) = e2e_functions.make_csv_data(self._csv_filename, 100,
                                              'regression')

    with open(self._schema_filename, 'w') as f:
      f.write(json.dumps(schema, indent=2, separators=(',', ': ')))

    e2e_functions.run_preprocess(output_dir=self._test_dir,
                                 csv_filename=self._csv_filename,
                                 schema_filename=self._schema_filename)

    metadata_path = os.path.join(self._test_dir, 'metadata.json')
    metadata = ml.features.FeatureMetadata.get_metadata(metadata_path)

    expected_features = {
        'num1': {'dtype': 'float', 'type': 'dense', 'name': 'num1',
                 'columns': ['num1'], 'size': 1},
        'num2': {'dtype': 'float', 'type': 'dense', 'name': 'num2',
                 'columns': ['num2'], 'size': 1},
        'num3': {'dtype': 'float', 'type': 'dense', 'name': 'num3',
                 'columns': ['num3'], 'size': 1},
        'str3': {'dtype': 'int64', 'type': 'sparse', 'name': 'str3',
                 'columns': ['str3'], 'size': 7},
        'str2': {'dtype': 'int64', 'type': 'sparse', 'name': 'str2',
                 'columns': ['str2'], 'size': 7},
        'str1': {'dtype': 'int64', 'type': 'sparse', 'name': 'str1',
                 'columns': ['str1'], 'size': 8},
        'key': {'dtype': 'bytes', 'type': 'dense', 'name': 'key',
                'columns': ['key'], 'size': 1},
        'target': {'dtype': 'float', 'type': 'dense', 'name': 'target',
                   'columns': ['target'], 'size': 1}}

    self.assertEqual(metadata.features, expected_features)
    self.assertEqual(metadata.columns['target']['scenario'], 'continuous')
    train_files = glob.glob(os.path.join(self._test_dir, 'features_train*'))
    self.assertTrue(train_files)
    self.assertTrue(os.path.isfile(os.path.join(self._test_dir, 'schema.json')))

    # Inspect the first TF record.
    for line in tf.python_io.tf_record_iterator(train_files[0],
        options=tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.GZIP)):
      ex = tf.train.Example()
      ex.ParseFromString(line)
      self.assertTrue('num1' in ex.features.feature)
      self.assertTrue('num2' in ex.features.feature)
      self.assertTrue('num3' in ex.features.feature)
      self.assertTrue('key' in ex.features.feature)
      self.assertTrue('target' in ex.features.feature)
      self.assertTrue('str1@0' in ex.features.feature)
      self.assertTrue('str1@1' in ex.features.feature)
      self.assertTrue('str2@0' in ex.features.feature)
      self.assertTrue('str2@1' in ex.features.feature)
      self.assertTrue('str3@0' in ex.features.feature)
      self.assertTrue('str3@1' in ex.features.feature)
      break

  def testClassification(self):
    (schema, _) = e2e_functions.make_csv_data(self._csv_filename, 100,
                                              'classification')
    with open(self._schema_filename, 'w') as f:
      f.write(json.dumps(schema, indent=2, separators=(',', ': ')))

    e2e_functions.run_preprocess(output_dir=self._test_dir,
                                 csv_filename=self._csv_filename,
                                 schema_filename=self._schema_filename,
                                 train_percent='90',
                                 eval_percent='10',
                                 test_percent='0')

    metadata_path = os.path.join(self._test_dir, 'metadata.json')
    metadata = ml.features.FeatureMetadata.get_metadata(metadata_path)

    self.assertEqual(metadata.columns['target']['scenario'], 'discrete')
    self.assertTrue(glob.glob(os.path.join(self._test_dir, 'features_train*')))
    self.assertTrue(glob.glob(os.path.join(self._test_dir, 'features_eval*')))
    self.assertFalse(glob.glob(os.path.join(self._test_dir, 'features_test*')))

if __name__ == '__main__':
    unittest.main()