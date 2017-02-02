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
  """Tests training.

  Each test builds a csv test dataset. Preprocessing is run on the data to
  produce analysis. Training is then ran, and the output is collected and the
  accuracy/loss values are inspected.
  """

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

    self._transforms_filename = os.path.join(self._test_dir, 'transforms.json')


  def tearDown(self):
    print('TestTrainer: removing test dir ' + self._test_dir)
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
    e2e_functions.make_preprocess_schema(self._schema_filename)
    e2e_functions.make_preprocess_input_features(self._input_features_filename, 
                                                 problem_type)

    e2e_functions.run_preprocess(
        output_dir=self._preprocess_output,
        csv_filename=self._csv_train_filename,
        schema_filename=self._schema_filename,
        input_features_filename=self._input_features_filename)

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
        max_steps=2500,
        model_type=model_type + '_' + problem_type,
        extra_args=extra_args)
    self._training_screen_output = output
    #print(self._training_screen_output)
    

  def _check_training_screen_output(self, accuracy=None, loss=None):
    """Should be called after _run_training.

    Inspects self._training_screen_output for correct output.

    Args:
      accuracy: float. Eval accuracy should be > than this number.
      loss: flaot. Eval loss should be < than this number.
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
    model_folder = os.path.join(self._train_output, 'model')
    self.assertTrue(os.path.isfile(os.path.join(model_folder, 'checkpoint')))
    self.assertTrue(os.path.isfile(os.path.join(model_folder, 'export')))
    self.assertTrue(os.path.isfile(os.path.join(model_folder, 'export.meta')))
    self.assertTrue(os.path.isfile(os.path.join(model_folder, 'schema.json')))
    self.assertTrue(os.path.isfile(os.path.join(model_folder, 'transforms.json')))
    self.assertTrue(os.path.isfile(os.path.join(model_folder, 'input_features.json')))


  def testRegressionDnn(self):
    print('\n\nTesting Regression DNN')

    transforms = {
        "num1": {"transform": "scale"},
        "num2": {"transform": "max_abs_scale","value": 4},
        "str1": {"transform": "hash_embedding", "embedding_dim": 2, "hash_bucket_size": 4},
        "str2": {"transform": "embedding", "embedding_dim": 3},
        "target": {"transform": "target"}
    }

    extra_args = ['--layer_sizes 10 10 5']
    self._run_training(problem_type='regression', 
                       model_type='dnn',
                       transforms=transforms,
                       extra_args=extra_args)

    self._check_training_screen_output(loss=6)
    self._check_train_files()


  def testRegressionLinear(self):
    print('\n\nTesting Regression Linear')

    transforms = {
        "num1": {"transform": "scale"},
        "num2": {"transform": "max_abs_scale","value": 4},
        "str1": {"transform": "hash_sparse", "hash_bucket_size": 2},
        "str2": {"transform": "hash_sparse", "hash_bucket_size": 2},
        "str3": {"transform": "hash_sparse", "hash_bucket_size": 2},
        "target": {"transform": "target"}
    }

    self._run_training(problem_type='regression', 
                       model_type='linear',
                       transforms=transforms)

    self._check_training_screen_output(loss=100)
    self._check_train_files()


  def testClassificationDnn(self):
    print('\n\nTesting classification DNN')

    transforms = {
        "num1": {"transform": "scale"},
        "num2": {"transform": "max_abs_scale","value": 4},
        "str1": {"transform": "hash_one_hot", "hash_bucket_size": 4},
        "str2": {"transform": "one_hot"},
        "str3": {"transform": "embedding", "embedding_dim": 3},
        "target": {"transform": "target"}
    }

    extra_args = ['--layer_sizes 10 10 5']
    self._run_training(problem_type='classification', 
                       model_type='dnn',
                       transforms=transforms,
                       extra_args=extra_args)

    self._check_training_screen_output(accuracy=0.95, loss=0.09)
    self._check_train_files()


  def testClassificationLinear(self):
    print('\n\nTesting classification Linear')

    transforms = {
        "num1": {"transform": "scale"},
        "num2": {"transform": "max_abs_scale","value": 4},
        "str1": {"transform": "hash_sparse", "hash_bucket_size": 4},
        "str2": {"transform": "sparse"},
        "target": {"transform": "target"}
    }

    self._run_training(problem_type='classification', 
                       model_type='linear',
                       transforms=transforms)

    self._check_training_screen_output(accuracy=0.90, loss=0.2)
    self._check_train_files()

