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
import filecmp
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
    self._input_features_filename = os.path.join(self._test_dir, 
                                                 'input_features_file.json')

    self._preprocess_output = os.path.join(self._test_dir, 'pout')

  def tearDown(self):
    print('TestPreprocess: removing test dir: ' + self._test_dir)
    shutil.rmtree(self._test_dir)

  
  def _make_test_data(self, problem_type):
    """Makes input files to run preprocessing on.

    Args:
      problem_type: 'regression' or 'classification'
    """
    e2e_functions.make_csv_data(self._csv_filename, 100, problem_type, True)
    e2e_functions.make_preprocess_schema(self._schema_filename)
    e2e_functions.make_preprocess_input_features(self._input_features_filename, 
                                                 problem_type)

  def _test_preprocess(self, problem_type):
    self._make_test_data(problem_type)

    e2e_functions.run_preprocess(
        output_dir=self._preprocess_output,
        csv_filename=self._csv_filename,
        schema_filename=self._schema_filename,
        input_features_filename=self._input_features_filename)


    schema_file = os.path.join(self._preprocess_output, 'schema.json')
    features_file = os.path.join(self._preprocess_output, 'input_features.json')
    numerical_analysis_file = os.path.join(self._preprocess_output, 'numerical_analysis.json')

    # test schema and features were copied
    self.assertTrue(filecmp.cmp(schema_file, self._schema_filename))
    self.assertTrue(filecmp.cmp(features_file, self._input_features_filename))

    expected_numerical_keys = ['num1', 'num2', 'num3']
    if problem_type == 'regression':
      expected_numerical_keys.append('target')

    # Load the numerical analysis file and check it has the right keys
    with open(numerical_analysis_file, 'r') as f:
      analysis = json.load(f)
    self.assertEqual(sorted(expected_numerical_keys), sorted(analysis.keys()))

    # Check that the vocab files are made
    expected_vocab_files = ['vocab_str1.csv', 'vocab_str2.csv', 'vocab_str3.csv']
    if problem_type == 'classification':
      expected_vocab_files.append('vocab_target.csv')

    for name in expected_vocab_files:
      vocab_file = os.path.join(self._preprocess_output, name)
      self.assertTrue(os.path.exists(vocab_file))
      self.assertGreater(os.path.getsize(vocab_file), 0)

    all_expected_files = (expected_vocab_files + ['input_features.json',
                          'numerical_analysis.json', 'schema.json'])
    all_file_paths = glob.glob(os.path.join(self._preprocess_output, '*'))
    all_files = [os.path.basename(path) for path in all_file_paths]
    self.assertEqual(sorted(all_expected_files), sorted(all_files))


  def testRegression(self):
    self._test_preprocess('regression')

  def testClassification(self):
    self._test_preprocess('classification')

if __name__ == '__main__':
    unittest.main()
