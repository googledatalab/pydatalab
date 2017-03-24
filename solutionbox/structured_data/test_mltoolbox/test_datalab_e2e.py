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
"""Test analyze, training, and prediction.
"""
from __future__ import absolute_import
from __future__ import print_function

import json
import logging
import os
import pandas as pd
import shutil
import six
import sys
import tempfile
import unittest

from . import e2e_functions
from tensorflow.python.lib.io import file_io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../..')))

import mltoolbox.regression.linear as reglinear  # noqa: E402
import google.datalab.ml as dlml  # noqa: E402


class TestLinearRegression(unittest.TestCase):
  """Test linear regression works e2e locally.

  Note that there should be little need for testing the other scenarios (linear
  classification, dnn regression, dnn classification) as they should only
  differ at training time. The training coverage of task.py is already done in
  test_sd_trainer.
  """
  def __init__(self, *args, **kwargs):
    super(TestLinearRegression, self).__init__(*args, **kwargs)

    # Log everything
    self._logger = logging.getLogger('TestStructuredDataLogger')
    self._logger.setLevel(logging.DEBUG)
    if not self._logger.handlers:
      self._logger.addHandler(logging.StreamHandler(stream=sys.stdout))

  def _make_test_files(self):
    """Builds test files and folders"""

    # Make the output folders
    self._test_dir = tempfile.mkdtemp()
    self._preprocess_output = os.path.join(self._test_dir, 'preprocess')
    self._train_output = os.path.join(self._test_dir, 'train')
    self._batch_predict_output = os.path.join(self._test_dir, 'batch_predict')

    # Don't make train_output folder as it should not exist at training time.
    os.mkdir(self._preprocess_output)
    os.mkdir(self._batch_predict_output)

    # Make csv files
    self._csv_train_filename = os.path.join(self._test_dir,
                                            'train_csv_data.csv')
    self._csv_eval_filename = os.path.join(self._test_dir,
                                           'eval_csv_data.csv')
    self._csv_predict_filename = os.path.join(self._test_dir,
                                              'predict_csv_data.csv')
    e2e_functions.make_csv_data(self._csv_train_filename, 100, 'regression',
                                True)
    e2e_functions.make_csv_data(self._csv_eval_filename, 100, 'regression',
                                True)
    self._predict_num_rows = 10
    e2e_functions.make_csv_data(self._csv_predict_filename,
                                self._predict_num_rows, 'regression', False)

    # Make schema file
    self._schema_filename = os.path.join(self._test_dir, 'schema.json')
    e2e_functions.make_preprocess_schema(self._schema_filename, 'regression')

    # Make feature file
    self._input_features_filename = os.path.join(self._test_dir,
                                                 'input_features_file.json')
    transforms = {
        "num1": {"transform": "scale"},
        "num2": {"transform": "scale", "value": 4},
        "str1": {"transform": "one_hot"},
        "str2": {"transform": "embedding", "embedding_dim": 3},
        "target": {"transform": "target"},
        "key": {"transform": "key"},
    }
    file_io.write_string_to_file(
        self._input_features_filename,
        json.dumps(transforms, indent=2))

  def _run_analyze(self):
    reglinear.analyze(
        output_dir=self._preprocess_output,
        dataset=dlml.CsvDataSet(
            file_pattern=self._csv_train_filename,
            schema_file=self._schema_filename))

    self.assertTrue(os.path.isfile(
        os.path.join(self._preprocess_output, 'stats.json')))
    self.assertTrue(os.path.isfile(
        os.path.join(self._preprocess_output, 'vocab_str1.csv')))

  def _run_train(self):
    reglinear.train(
        train_dataset=dlml.CsvDataSet(
            file_pattern=self._csv_train_filename,
            schema_file=self._schema_filename),
        eval_dataset=dlml.CsvDataSet(
            file_pattern=self._csv_eval_filename,
            schema_file=self._schema_filename),
        analysis_dir=self._preprocess_output,
        output_dir=self._train_output,
        features=self._input_features_filename,
        max_steps=100,
        train_batch_size=100)

    self.assertTrue(os.path.isfile(
        os.path.join(self._train_output, 'model', 'saved_model.pb')))
    self.assertTrue(os.path.isfile(
        os.path.join(self._train_output, 'evaluation_model', 'saved_model.pb')))

  def _run_predict(self):
    data = pd.read_csv(self._csv_predict_filename,
                       header=None)
    df = reglinear.predict(data=data,
                           training_dir=self._train_output)

    self.assertEqual(len(df.index), self._predict_num_rows)
    self.assertEqual(list(df), ['key', 'predicted'])

  def _run_batch_prediction(self, output_dir, use_target):
    reglinear.batch_predict(
        training_dir=self._train_output,
        prediction_input_file=(self._csv_eval_filename if use_target
                               else self._csv_predict_filename),
        output_dir=output_dir,
        mode='evaluation' if use_target else 'prediction',
        batch_size=4,
        output_format='csv')

    # check errors file is empty
    errors = file_io.get_matching_files(os.path.join(output_dir, 'errors*'))
    self.assertEqual(len(errors), 1)
    self.assertEqual(os.path.getsize(errors[0]), 0)

    # check predictions files are not empty
    predictions = file_io.get_matching_files(os.path.join(output_dir,
                                                          'predictions*'))
    self.assertGreater(os.path.getsize(predictions[0]), 0)

    # check the schema is correct
    schema_file = os.path.join(output_dir, 'csv_schema.json')
    self.assertTrue(os.path.isfile(schema_file))
    schema = json.loads(file_io.read_file_to_string(schema_file))
    self.assertEqual(schema[0]['name'], 'key')
    self.assertEqual(schema[1]['name'], 'predicted')
    if use_target:
      self.assertEqual(schema[2]['name'], 'target')
      self.assertEqual(len(schema), 3)
    else:
      self.assertEqual(len(schema), 2)

  def _cleanup(self):
    shutil.rmtree(self._test_dir)

  def test_e2e(self):
    try:
      self._make_test_files()
      self._run_analyze()
      if six.PY2:
        self._run_train()
        self._run_predict()
        self._run_batch_prediction(
            os.path.join(self._batch_predict_output, 'with_target'),
            True)
        self._run_batch_prediction(
            os.path.join(self._batch_predict_output, 'without_target'),
            False)
      else:
        print('only tested analyze in TestLinearRegression')
    finally:
      self._cleanup()


if __name__ == '__main__':
    unittest.main()
