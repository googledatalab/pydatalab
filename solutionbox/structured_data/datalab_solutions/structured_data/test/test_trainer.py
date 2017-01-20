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


class TestTrainer(unittest.TestCase):
  def setUp(self):
    self._test_dir = tempfile.mkdtemp()
    self._preprocess_dir = os.path.join(self._test_dir, 'pre')
    self._train_dir = os.path.join(self._test_dir, 'train')

    os.mkdir(self._preprocess_dir)
    os.mkdir(self._train_dir)

    self._config_filename = os.path.join(self._preprocess_dir, 'config.json')
    self._csv_filename = os.path.join(self._preprocess_dir, 'raw_csv_data.csv')

    
  def tearDown(self):
    shutil.rmtree(self._test_dir)


  def _run_training(self, config):
    with open(self._config_filename, 'w') as f:
      f.write(json.dumps(config, indent=2, separators=(',', ': ')))
    
    e2e_functions.run_preprocess(self._preprocess_dir, self._csv_filename, self._config_filename)
    e2e_functions.run_training(self._train_dir, self._preprocess_dir, self._config_filename, ['--layer_sizes 20 10 5'])

  def _check_train_files(self):
    model_folder = os.path.join(self._train_dir, 'model')
    self.assertTrue(os.path.isfile(os.path.join(model_folder, 'checkpoint')))
    self.assertTrue(os.path.isfile(os.path.join(model_folder, 'export')))
    self.assertTrue(os.path.isfile(os.path.join(model_folder, 'export.meta')))
    self.assertTrue(os.path.isfile(os.path.join(model_folder, 'metadata.json')))


  def testRegressionDnn(self):
    print('\n\nTesting Regression DNN')
    config = e2e_functions.make_csv_data(self._csv_filename, 5000, 'regression')
    config['categorical']['str1']['transform'] = 'embedding'
    config['categorical']['str1']['dimension'] = '3'
    config['model_type'] = 'dnn'

    self._run_training(config)
    self._check_train_files()


  def testRegressionLinear(self):
    print('\n\nTesting Regression Linear')
    config = e2e_functions.make_csv_data(self._csv_filename, 5000, 'regression')
    config['model_type'] = 'linear'

    self._run_training(config)
    self._check_train_files()


  def testClassificationDnn(self):
    print('\n\nTesting classification DNN')
    config = e2e_functions.make_csv_data(self._csv_filename, 5000,
                                         'classification')
    config['categorical']['str1']['transform'] = 'embedding'
    config['categorical']['str1']['dimension'] = '3'
    config['model_type'] = 'dnn'

    self._run_training(config)
    self._check_train_files()


  def testClassificationLinear(self):
    print('\n\nTesting classification Linear')
    config = e2e_functions.make_csv_data(self._csv_filename, 5000,
                                         'classification')      
    config['model_type'] = 'linear'

    self._run_training(config)
    self._check_train_files()