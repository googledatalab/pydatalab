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
from __future__ import absolute_import

import json
import logging
import os
import re
import shutil
import sys
import tempfile
import unittest

from . import e2e_functions


class TestTrainer(unittest.TestCase):
  """Tests training.

  Each test builds a csv test dataset. Preprocessing is run on the data to
  produce analysis. Training is then ran, and the output is collected and the
  accuracy/loss values are inspected.
  """
  def __init__(self, *args, **kwargs):
    super(TestTrainer, self).__init__(*args, **kwargs)

    # Allow this class to be subclassed for quick tests that only care about
    # training working, not model loss/accuracy.
    self._max_steps = 2500
    self._check_model_fit = True

    # Log everything
    self._logger = logging.getLogger('TestStructuredDataLogger')
    self._logger.setLevel(logging.DEBUG)
    if not self._logger.handlers:
      self._logger.addHandler(logging.StreamHandler(stream=sys.stdout))

  def setUp(self):
    self._test_dir = tempfile.mkdtemp()
    self._preprocess_output = os.path.join(self._test_dir, 'pre')
    self._train_output = os.path.join(self._test_dir, 'train')

    os.mkdir(self._preprocess_output)
    os.mkdir(self._train_output)

    self._csv_train_filename = os.path.join(self._test_dir, 'train_csv_data.csv')
    self._csv_eval_filename = os.path.join(self._test_dir, 'eval_csv_data.csv')
    self._schema_filename = os.path.join(self._test_dir, 'schema.json')
    self._input_features_filename = os.path.join(self._test_dir,
                                                 'input_features_file.json')

    self._transforms_filename = os.path.join(self._test_dir, 'features.json')

  def tearDown(self):
    self._logger.debug('TestTrainer: removing test dir ' + self._test_dir)
    shutil.rmtree(self._test_dir)

  def _run_training(self, problem_type, model_type, transforms, extra_args=[]):
    """Runs training.

    Output is saved to _training_screen_output. Nothing from training should be
    printed to the screen.

    Args:
      problem_type: 'regression' or 'classification'
      model_type: 'linear' or 'dnn'
      transform: JSON object of the transforms file.
      extra_args: list of strings to pass to the trainer.
    """
    # Run preprocessing.
    e2e_functions.make_csv_data(self._csv_train_filename, 100, problem_type, True)
    e2e_functions.make_csv_data(self._csv_eval_filename, 100, problem_type, True)
    e2e_functions.make_preprocess_schema(self._schema_filename, problem_type)

    e2e_functions.run_preprocess(
        output_dir=self._preprocess_output,
        csv_filename=self._csv_train_filename,
        schema_filename=self._schema_filename,
        logger=self._logger)

    # Write the transforms file.
    with open(self._transforms_filename, 'w') as f:
      f.write(json.dumps(transforms, indent=2, separators=(',', ': ')))

    # Run training and save the output.
    output = e2e_functions.run_training(
        train_data_paths=self._csv_train_filename,
        eval_data_paths=self._csv_eval_filename,
        output_path=self._train_output,
        preprocess_output_dir=self._preprocess_output,
        transforms_file=self._transforms_filename,
        max_steps=self._max_steps,
        model_type=model_type + '_' + problem_type,
        logger=self._logger,
        extra_args=extra_args)
    self._training_screen_output = output

  def _check_training_screen_output(self, accuracy=None, loss=None):
    """Should be called after _run_training.

    Inspects self._training_screen_output for correct output.

    Args:
      accuracy: float. Eval accuracy should be > than this number.
      loss: flaot. Eval loss should be < than this number.
    """
    if not self._check_model_fit:
      self._logger.debug('Skipping model loss/accuracy checks')
      return

    # Find the last training loss line in the output
    lines = self._training_screen_output.splitlines()
    last_line = None
    for line in lines:
      if line.startswith('INFO:tensorflow:Saving dict for global step %d' % self._max_steps):
        last_line = line
        break

    if not last_line:
      self._logger.debug('Skipping _check_training_screen_output as could not '
                         'find last eval line')
      return
    self._logger.debug(last_line)

    # supports positive numbers (int, real) with exponential form support.
    positive_number_re = re.compile('[+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')

    # Check it made it to step 2500
    saving_num_re = re.compile('global_step = \d+')
    saving_num = saving_num_re.findall(last_line)
    # saving_num == ['Saving evaluation summary for step NUM']
    self.assertEqual(len(saving_num), 1)
    step_num = positive_number_re.findall(saving_num[0])
    # step_num == ['2500']
    self.assertEqual(len(step_num), 1)
    self.assertEqual(int(step_num[0]), self._max_steps)

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
    self._check_savedmodel(os.path.join(self._train_output, 'model'))
    self._check_savedmodel(os.path.join(self._train_output, 'evaluation_model'))

  def _check_savedmodel(self, model_folder):
    self.assertTrue(
        os.path.isfile(os.path.join(model_folder, 'saved_model.pb')))
    self.assertTrue(
        os.path.isfile(os.path.join(model_folder, 'variables/variables.index')))
    self.assertTrue(
        os.path.isfile(os.path.join(model_folder, 'assets.extra/schema.json')))
    self.assertTrue(
        os.path.isfile(os.path.join(model_folder, 'assets.extra/features.json')))

  def testRegressionDnn(self):
    self._logger.debug('\n\nTesting Regression DNN')

    transforms = {
        "num1": {"transform": "scale"},
        "num2": {"transform": "scale", "value": 4},
        "str1": {"transform": "one_hot"},
        "str2": {"transform": "embedding", "embedding_dim": 3},
        "target": {"transform": "target"},
        "key": {"transform": "key"},
    }

    extra_args = ['--layer-size1=10', '--layer-size2=10', '--layer-size3=5']
    self._run_training(problem_type='regression',
                       model_type='dnn',
                       transforms=transforms,
                       extra_args=extra_args)

    self._check_training_screen_output(loss=20)
    self._check_train_files()

  def testRegressionLinear(self):
    self._logger.debug('\n\nTesting Regression Linear')

    transforms = {
        "num1": {"transform": "scale"},
        "num2": {"transform": "scale", "value": 4},
        "str1": {"transform": "one_hot"},
        "str2": {"transform": "embedding", "embedding_dim": 3},
        "target": {"transform": "target"},
        "key": {"transform": "key"},
    }

    self._run_training(problem_type='regression',
                       model_type='linear',
                       transforms=transforms)

    self._check_training_screen_output(loss=20)
    self._check_train_files()

  def testClassificationDnn(self):
    self._logger.debug('\n\nTesting classification DNN')

    transforms = {
        "num1": {"transform": "scale"},
        "num2": {"transform": "scale", "value": 4},
        "str1": {"transform": "one_hot"},
        "str2": {"transform": "embedding", "embedding_dim": 3},
        "target": {"transform": "target"},
        "key": {"transform": "key"},
    }

    extra_args = ['--layer-size1=10', '--layer-size2=10', '--layer-size3=5']
    self._run_training(problem_type='classification',
                       model_type='dnn',
                       transforms=transforms,
                       extra_args=extra_args)

    self._check_training_screen_output(accuracy=0.70, loss=0.10)
    self._check_train_files()

  def testClassificationLinear(self):
    self._logger.debug('\n\nTesting classification Linear')

    transforms = {
        "num1": {"transform": "scale"},
        "num2": {"transform": "scale", "value": 4},
        "str1": {"transform": "one_hot"},
        "str2": {"transform": "embedding", "embedding_dim": 3},
        "target": {"transform": "target"},
        "key": {"transform": "key"},
    }

    self._run_training(problem_type='classification',
                       model_type='linear',
                       transforms=transforms)

    self._check_training_screen_output(accuracy=0.70, loss=0.2)
    self._check_train_files()


if __name__ == '__main__':
    unittest.main()
