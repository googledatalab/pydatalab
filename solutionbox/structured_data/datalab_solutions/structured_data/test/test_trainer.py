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
    print('Removing temp dir ' + self._test_dir)
    shutil.rmtree(self._test_dir)

  def _run_training(self, schema, transforms, extra_args):
    with open(self._schema_filename, 'w') as f:
      f.write(json.dumps(schema, indent=2, separators=(',', ': ')))

    with open(self._transforms_filename, 'w') as f:
      f.write(json.dumps(transforms, indent=2, separators=(',', ': ')))


    e2e_functions.run_preprocess(output_dir=self._preprocess_dir,
                                 csv_filename=self._csv_filename,
                                 schema_filename=self._schema_filename)
    e2e_functions.run_training(output_dir=self._train_dir,
                               input_dir=self._preprocess_dir,
                               schema_filename=self._schema_filename,
                               transforms_filename=self._transforms_filename,
                               extra_args=extra_args)

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
             '--model_type=dnn',
             '--problem_type=regression']

    self._run_training(schema, transforms, flags)
    self._check_train_files()

  def testRegressionLinear(self):
    print('\n\nTesting Regression Linear')
    (schema, transforms) = e2e_functions.make_csv_data(self._csv_filename, 5000,
                                                       'regression')
    flags = ['--model_type=linear',
             '--problem_type=regression']

    self._run_training(schema, transforms, flags)
    self._check_train_files()

  def testClassificationDnn(self):
    print('\n\nTesting classification DNN')
    (schema, transforms) = e2e_functions.make_csv_data(self._csv_filename, 5000,
                                                       'classification')
    transforms['str1']['transform'] = 'embedding'
    transforms['str1']['dimension'] = '3'

    flags = ['--layer_sizes 10 10 5',
             '--model_type=dnn',
             '--problem_type=classification']

    self._run_training(schema, transforms, flags)
    self._check_train_files()

  def testClassificationLinear(self):
    print('\n\nTesting classification Linear')
    (schema, transforms) = e2e_functions.make_csv_data(self._csv_filename, 5000,
                                                       'classification')
    flags = ['--model_type=linear',
             '--problem_type=classification']

    self._run_training(schema, transforms, flags)
    self._check_train_files()
