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
import json
import os
import re
import shutil
import tempfile
import unittest

import e2e_functions


class TestTrainer(unittest.TestCase):

  def setUp(self):
    self._test_dir = tempfile.mkdtemp()
    self._preprocess_dir = os.path.join(self._test_dir, 'pre')
    self._train_dir = os.path.join(self._test_dir, 'train')

    os.mkdir(self._preprocess_dir)
    os.mkdir(self._train_dir)

    self._csv_filename = os.path.join(self._preprocess_dir, 'raw_csv_data.csv')
    self._schema_filename = os.path.join(self._test_dir, 'schema.json')
    self._transforms_filename = os.path.join(self._test_dir, 'transforms.json')

  def tearDown(self):
    print('TestTrainer: removing test dir ' + self._test_dir)
    #shutil.rmtree(self._test_dir)

  def _run_training(self, schema, transforms, extra_args):
    with open(self._schema_filename, 'w') as f:
      f.write(json.dumps(schema, indent=2, separators=(',', ': ')))

    with open(self._transforms_filename, 'w') as f:
      f.write(json.dumps(transforms, indent=2, separators=(',', ': ')))


    e2e_functions.run_preprocess(output_dir=self._preprocess_dir,
                                 csv_filename=self._csv_filename,
                                 schema_filename=self._schema_filename)
    output = e2e_functions.run_training(output_dir=self._train_dir,
                               input_dir=self._preprocess_dir,
                               schema_filename=self._schema_filename,
                               transforms_filename=self._transforms_filename,
                               max_steps=2500,
                               extra_args=extra_args)
    self._training_screen_output = output

  def _check_training_screen_output(self, accuracy=None, loss=None):
    """Should be called after _run_training.

    Inspects self._training_screen_output for correct output.

    Args:
      eval_metrics: dict in the form {key: expected_number}. Will inspect the
          last line of the training output for the line "KEY = NUMBER" and
          check that NUMBER < expected_number.
    """
    # Print the last line of training output which has the loss value.
    lines = self._training_screen_output.splitlines()
    last_line = lines[len(lines)-1]
    print(last_line)

    # supports positive numbers (int, real) with exponential form support.
    positive_number_re = re.compile('[+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')

    # Check it made it to step 2500
    saving_num_re = re.compile('Saving evaluation summary for step \d+')
    saving_num = saving_num_re.findall(last_line)
    # saving_num == ['Saving evaluation summary for step NUM']
    self.assertEqual(len(saving_num), 1)
    step_num = positive_number_re.findall(saving_num[0])
    # step_num == ['2500']
    self.assertEqual(len(step_num), 1)
    self.assertEqual(int(step_num[0]), 2500)


    # Check the accuracy
    if accuracy is not None:
      accuracy_eq_num_re = re.compile('accuracy = [+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')
      accuracy_eq_num = accuracy_eq_num_re.findall(last_line)
      # accuracy_eq_num == ['accuracy = NUM']
      self.assertEqual(len(accuracy_eq_num), 1)
      accuracy_num = positive_number_re.findall(accuracy_eq_num[0])
      # accuracy_num == ['X.XXX']
      self.assertEqual(len(accuracy_num), 1)
      self.assertGreater(float(accuracy_num[0]), accuracy)

    if loss is not None:
      loss_eq_num_re = re.compile('loss = [+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')
      loss_eq_num = loss_eq_num_re.findall(last_line)
      # loss_eq_num == ['loss = NUM']
      self.assertEqual(len(loss_eq_num), 1)
      loss_num = positive_number_re.findall(loss_eq_num[0])
      # loss_num == ['X.XXX']
      self.assertEqual(len(loss_num), 1)
      self.assertLess(float(loss_num[0]), loss)




  def _check_train_files(self):
    model_folder = os.path.join(self._train_dir, 'model')
    self.assertTrue(os.path.isfile(os.path.join(model_folder, 'checkpoint')))
    self.assertTrue(os.path.isfile(os.path.join(model_folder, 'export')))
    self.assertTrue(os.path.isfile(os.path.join(model_folder, 'export.meta')))
    self.assertTrue(os.path.isfile(os.path.join(model_folder, 'metadata.json')))
    self.assertTrue(os.path.isfile(os.path.join(model_folder, 'schema.json')))
    self.assertTrue(os.path.isfile(os.path.join(model_folder, 'transforms.json')))

  def testRegressionDnn(self):
    print('\n\nTesting Regression DNN')
    (schema, transforms) = e2e_functions.make_csv_data(self._csv_filename, 5000,
                                                       'regression')
    transforms['str1']['transform'] = 'embedding'
    transforms['str1']['dimension'] = '3'

    flags = ['--layer_sizes 10 10 5',
             '--model_type=dnn_regression']

    self._run_training(schema, transforms, flags)
    self._check_training_screen_output(loss=10)
    self._check_train_files()

  def testRegressionLinear(self):
    print('\n\nTesting Regression Linear')
    (schema, transforms) = e2e_functions.make_csv_data(self._csv_filename, 5000,
                                                       'regression')
    flags = ['--model_type=linear_regression']

    self._run_training(schema, transforms, flags)
    self._check_training_screen_output(loss=1)
    self._check_train_files()

  def testClassificationDnn(self):
    print('\n\nTesting classification DNN')
    (schema, transforms) = e2e_functions.make_csv_data(self._csv_filename, 5000,
                                                       'classification')
    transforms['str1']['transform'] = 'embedding'
    transforms['str1']['dimension'] = '3'

    flags = ['--layer_sizes 10 10 5',
             '--model_type=dnn_classification']

    self._run_training(schema, transforms, flags)
    self._check_training_screen_output(accuracy=0.95, loss=0.09)
    self._check_train_files()

  def testClassificationLinear(self):
    print('\n\nTesting classification Linear')
    (schema, transforms) = e2e_functions.make_csv_data(self._csv_filename, 5000,
                                                       'classification')
    flags = ['--model_type=linear_classification']

    self._run_training(schema, transforms, flags)
    self._check_training_screen_output(accuracy=0.95, loss=0.15)
    self._check_train_files()
