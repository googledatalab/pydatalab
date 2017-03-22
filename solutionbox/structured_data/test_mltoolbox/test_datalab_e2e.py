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

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../..')))

import mltoolbox.regression.linear as reglinear  # noqa: E402
import google.datalab.ml as dlml  # noqa: E402
import unittest  # noqa: E402

from . import e2e_functions

class TestLinearRegression(unittest.TestCase):
  """Test linear regression works e2e locally.

  Note that there should be little need for testing the other scenarios (linear
  classification, dnn regression, dnn classification) as they should only 
  differ at training time. The training coverage is already done in
  test_sd_trainer.
  """

  def _make_test_files(self):
    """Builds test files and folders"""

    # Make the output folders
    self._test_dir = tempfile.mkdtemp()
    self._preprocess_output = os.path.join(self._test_dir, 'pre')
    self._train_output = os.path.join(self._test_dir, 'train')
    self._predict_output = os.path.join(self._test_dir, 'predict')
    self._batch_predict_output = os.path.join(self._test_dir, 'batch)predict')

    os.mkdir(self._preprocess_output)
    os.mkdir(self._train_output)
    os.mkdir(self._predict_output)
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
    e2e_functions.make_csv_data(self._csv_predict_filename, 10, 'regression', 
                                False)

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


  def test_e2e(self):
    self._make_test_files()
    self._run_analyze()
    self._run_train()
    self._run_predict()
    self._run_batch_prediction(True)  # Evaluation mode
    self._run_batch_prediction(False)  # Prediction mode