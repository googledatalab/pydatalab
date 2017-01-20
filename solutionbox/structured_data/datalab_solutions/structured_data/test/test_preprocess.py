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

import unittest
import tempfile
import os
import sys
import json
import glob
import subprocess
import shutil

import google.cloud.ml as ml

import e2e_functions


class TestPreprocess(unittest.TestCase):
  def setUp(self):
    self._test_dir = tempfile.mkdtemp()

    self._csv_filename = os.path.join(self._test_dir, 'raw_csv_data.csv')
    self._config_filename = os.path.join(self._test_dir, 'config.json')

  def tearDown(self):
    shutil.rmtree(self._test_dir)

  def testRegression(self):
    config = e2e_functions.make_csv_data(self._csv_filename, 100, 'regression')
    config['categorical']['str1']['transform'] = 'embedding'
    config['categorical']['str1']['dimension'] = '3'

    with open(self._config_filename, 'w') as f:
      f.write(json.dumps(config, indent=2, separators=(',', ': ')))
    
    e2e_functions.run_preprocess(self._test_dir, self._csv_filename, self._config_filename)

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
    self.assertTrue(glob.glob(os.path.join(self._test_dir, 'features_train*')))


  def testClassification(self):
    config = e2e_functions.make_csv_data(self._csv_filename, 100, 'classification')

    with open(self._config_filename, 'w') as f:
      f.write(json.dumps(config, indent=2, separators=(',', ': ')))
    
    e2e_functions.run_preprocess(self._test_dir, self._csv_filename, self._config_filename,
      '90', '10', '0')

    metadata_path = os.path.join(self._test_dir, 'metadata.json')
    metadata = ml.features.FeatureMetadata.get_metadata(metadata_path)

    self.assertEqual(metadata.columns['target']['scenario'], 'discrete')
    self.assertTrue(glob.glob(os.path.join(self._test_dir, 'features_train*')))
    self.assertTrue(glob.glob(os.path.join(self._test_dir, 'features_eval*')))
    self.assertFalse(glob.glob(os.path.join(self._test_dir, 'features_test*')))
