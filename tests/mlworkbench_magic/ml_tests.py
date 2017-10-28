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
"""Tests the \%\%ml magics functions without runing any jobs."""

from __future__ import absolute_import
from __future__ import unicode_literals
import unittest
import mock
import os


# import Python so we can mock the parts we need to here.
import IPython.core.display
import IPython.core.magic


def noop_decorator(func):
  return func


IPython.core.magic.register_line_cell_magic = noop_decorator
IPython.core.magic.register_line_magic = noop_decorator
IPython.core.magic.register_cell_magic = noop_decorator
IPython.core.display.HTML = lambda x: x
IPython.core.display.JSON = lambda x: x
IPython.get_ipython = mock.Mock()
IPython.get_ipython().user_ns = {}

import google.datalab.contrib.mlworkbench.commands._ml as mlmagic  # noqa


def find_key_value(arg_list, key, value):
  """Checks '--key value' is in arg_list."""
  for i in range(len(arg_list)):
    if arg_list[i] == key and arg_list[i + 1] == value:
      return True
  return False


def find_key_endswith(arg_list, key, value):
  """Checks '--key prefix_<value>' is in arg_list."""
  for i in range(len(arg_list)):
    if arg_list[i] == key and arg_list[i + 1].endswith(value):
      return True
  return False


def find_startswith_endswith(arg_list, key, value):
  """Checks '--<key>anything<value>' is in arg_list."""
  for i in range(len(arg_list)):
    if arg_list[i].startswith(key) and arg_list[i].endswith(value):
      return True
  return False


class TestMLMagic(unittest.TestCase):

  @mock.patch('google.datalab.contrib.mlworkbench._shell_process.run_and_monitor')
  @mock.patch('subprocess.Popen')  # Because of the trainer help menu
  def test_analyze_csv_local(self, popen_mock, run_and_monitor_mock):
    mlmagic.ml(
      line='dataset create',
      cell="""\
          format: csv
          train: ./taxi/train.csv
          eval: ./taxi/eval.csv
          name: taxi_data
          schema:
              - name: unique_key
                type: STRING
              - name: fare
                type: FLOAT"""
    )
    mlmagic.ml(
        line='analyze',
        cell="""\
            output: my_out_dir
            data: taxi_data
            features: dummy_features""")
    cmd_list = run_and_monitor_mock.call_args[0][0]
    # cmd_list = [u'python', u'analyze.py', u'--output', 'path/my_out_dir',
    #   u'--csv=path/file*.csv', u'--schema', u'/path/schema.json',
    #   u'--features', u'path/features.json']

    self.assertEqual('python', cmd_list[0])
    self.assertEqual('analyze.py', cmd_list[1])
    self.assertIn('--schema', cmd_list)
    self.assertIn('--features', cmd_list)
    self.assertTrue(find_key_endswith(cmd_list, '--output', 'my_out_dir'))
    self.assertTrue(find_startswith_endswith(cmd_list, '--csv=', 'train.csv'))

  @mock.patch('google.datalab.contrib.mlworkbench._shell_process.run_and_monitor')
  @mock.patch('subprocess.Popen')  # Because of the trainer help menu
  def test_transform_csv(self, popen_mock, run_and_monitor_mock):
    mlmagic.ml(
      line='dataset create',
      cell="""\
          format: csv
          train: ./taxi/train.csv
          eval: ./taxi/eval.csv
          name: taxi_data
          schema:
              - name: unique_key
                type: STRING
              - name: fare
                type: FLOAT"""
    )
    mlmagic.ml(
        line='transform --shuffle --cloud',
        cell="""\
            output: my_out_dir
            analysis: my_analyze_dir
            batch_size: 123
            data: taxi_data
            cloud_config:
              project_id: my_id
              num_workers: 987
              worker_machine_type: BLUE
              job_name: RED""")
    cmd_list = run_and_monitor_mock.call_args[0][0]
    # cmd_list = [u'python', u'transform.py', u'--output', 'path/my_out_dir',
    #   u'--analysis', 'path/my_analyze_dir', u'--prefix', 'my_prefix',
    #   u'--shuffle', u'--batch-size', '100', u'--csv=/path/file*.csv'
    #   ...
    self.assertEqual('python', cmd_list[0])
    self.assertEqual('transform.py', cmd_list[1])
    self.assertIn('--shuffle', cmd_list)

    self.assertTrue(find_key_endswith(cmd_list, '--output', 'my_out_dir'))
    self.assertTrue(find_key_endswith(cmd_list, '--analysis', 'my_analyze_dir'))
    self.assertTrue(find_key_value(cmd_list, '--prefix', 'train') or
                    find_key_value(cmd_list, '--prefix', 'eval'))
    self.assertTrue(find_key_value(cmd_list, '--batch-size', '123'))
    self.assertTrue(find_startswith_endswith(cmd_list, '--csv=', 'train.csv') or
                    find_startswith_endswith(cmd_list, '--csv=', 'eval.csv'))
    self.assertTrue(find_key_value(cmd_list, '--project-id', 'my_id'))
    self.assertTrue(find_key_value(cmd_list, '--num-workers', '987'))
    self.assertTrue(find_key_value(cmd_list, '--worker-machine-type', 'BLUE'))
    self.assertTrue(find_key_value(cmd_list, '--job-name', 'RED'))

  @mock.patch('google.datalab.contrib.mlworkbench.commands._ml._show_job_link')
  @mock.patch('google.datalab.ml.package_and_copy')
  @mock.patch('google.datalab.ml.Job.submit_training')
  @mock.patch('subprocess.Popen')  # Because of the trainer help menu
  def test_train_csv(self, popen_mock, submit_training_mock,
                     package_and_copy_mock, _show_job_link_mock):
    mlmagic.ml(
      line='dataset create',
      cell="""\
          format: transformed
          train: ./taxi/train_tfrecord.tar.gz
          eval: ./taxi/eval_tfrecord.tar.gz
          name: taxi_data_transformed"""
    )
    mlmagic.ml(
        line='train --cloud',
        cell="""\
            output: gs://my_out_dir
            analysis: my_analyze_dir
            data: $taxi_data_transformed
            model_args:
              key: value
            cloud_config:
              job_name: job1
              project_id: id""")
    job_request = submit_training_mock.call_args[0][0]

    cmd_list = job_request['args']

    self.assertEqual(job_request['project_id'], 'id')
    self.assertEqual(job_request['job_dir'], 'gs://my_out_dir')
    self.assertEqual(job_request['python_module'], 'trainer.task')
    self.assertEqual(job_request['package_uris'], ['gs://my_out_dir/staging/trainer.tar.gz'])

    self.assertTrue(find_key_value(cmd_list, '--job-dir', 'gs://my_out_dir'))
    self.assertTrue(find_key_endswith(cmd_list, '--analysis', 'my_analyze_dir'))
    self.assertTrue(find_startswith_endswith(cmd_list, '--train=', 'train_tfrecord.tar.gz'))
    self.assertTrue(find_startswith_endswith(cmd_list, '--eval=', 'eval_tfrecord.tar.gz'))
    self.assertTrue(find_key_value(cmd_list, '--key', 'value'))

  @mock.patch('google.datalab.contrib.mlworkbench.commands._ml._show_job_link')
  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.ml.Job.submit_batch_prediction')
  @mock.patch('subprocess.Popen')  # Because of the trainer help menu
  def test_batch_predict_csv(self, popen_mock, submit_batch_prediction_mock,
                             default_mock, _show_job_link_mock):
    default_mock.return_value = mock.Mock(project_id='my_project_id')

    mlmagic.ml(
        line='batch_predict --cloud',
        cell="""\
            model: my_model.my_version
            output: gs://output
            format: json
            batch_size: 10
            data:
              csv: %s""" % os.path.abspath(__file__))

    job_args = submit_batch_prediction_mock.call_args[0][0]

    self.assertEqual(job_args['input_paths'], [os.path.abspath(__file__)])
    self.assertEqual(
        job_args['version_name'],
        'projects/my_project_id/models/my_model/versions/my_version')
    self.assertEqual(job_args['output_path'], 'gs://output')
    self.assertEqual(job_args['data_format'], 'TEXT')
