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
"""Tests the %%ml magics functions without runing any jobs."""

from __future__ import absolute_import
from __future__ import unicode_literals
import unittest

import csv
import mock
import os
import pandas as pd
import random
import shutil
import tempfile

import google.datalab
from google.datalab.ml import CsvDataSet

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

import google.datalab.contrib.mlworkbench.commands._ml as mlmagic

def find_key_value(arg_list, key, value):
  for i in range(len(arg_list)):
    if arg_list[i] == key and arg_list[i + 1] == value:
      return True
  return False

def find_key_endswith(arg_list, key, value):
  for i in range(len(arg_list)):
    if arg_list[i] == key and arg_list[i + 1].endswith(value):
      return True
  return False

def find_startswith_endswith(arg_list, key, value):
  for i in range(len(arg_list)):
    if arg_list[i].startswith(key) and arg_list[i].endswith(value):
      return True
  return False

class TestMLMagic(unittest.TestCase):

  @mock.patch('google.datalab.contrib.mlworkbench._shell_process.run_and_monitor')
  @mock.patch('subprocess.Popen')  # Because of the trainer help menu
  def test_analyze_csv_local(self, popen_mock, run_and_monitor_mock):
    mlmagic.ml(
        line='analyze',
        cell="""\
            output: my_out_dir
            training_data:
              csv: file*.csv
              schema: dummy_schema
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
    self.assertTrue(find_startswith_endswith(cmd_list, '--csv=', 'file*.csv'))

  @mock.patch('google.datalab.contrib.mlworkbench._shell_process.run_and_monitor')
  @mock.patch('subprocess.Popen')  # Because of the trainer help menu
  def test_transform_csv(self, popen_mock, run_and_monitor_mock):
    mlmagic.ml(
        line='transform --prefix my_prefix --shuffle --cloud',
        cell="""\
            output: my_out_dir
            analysis: my_analyze_dir
            batch_size: 123
            training_data:
              csv: file*.csv
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
    self.assertTrue(find_key_value(cmd_list, '--prefix', 'my_prefix'))
    self.assertTrue(find_key_value(cmd_list, '--batch-size', '123'))
    self.assertTrue(find_startswith_endswith(cmd_list, '--csv=', 'file*.csv'))
    self.assertTrue(find_key_value(cmd_list, '--project-id', 'my_id'))
    self.assertTrue(find_key_value(cmd_list, '--num-workers', '987'))
    self.assertTrue(find_key_value(cmd_list, '--worker-machine-type', 'BLUE'))
    self.assertTrue(find_key_value(cmd_list, '--job-name', 'RED'))

  @mock.patch('google.datalab.contrib.mlworkbench.commands._ml._show_job_link') 
  @mock.patch('google.datalab.ml.package_and_copy') 
  @mock.patch('google.datalab.ml.Job.submit_training') 
  @mock.patch('subprocess.Popen')  # Because of the trainer help menu
  def test_train_csv_local(self, popen_mock, run_and_monitor_mock,
                           package_and_copy_mock,_show_job_link_mock):
    mlmagic.ml(
        line='train --cloud',
        cell="""\
            output: gs://my_out_dir
            analysis: my_analyze_dir
            training_data:
              csv: file*.csv
            evaluation_data:
              transformed: file*.tfrecords.gz
            model_args:
              key: value
            cloud_config:
              job_name: job1
              project_id: id""")
    job_request = run_and_monitor_mock.call_args[0][0]

    cmd_list = job_request['args']

    self.assertEqual(job_request['project_id'], 'id')
    self.assertEqual(job_request['job_dir'], 'gs://my_out_dir')
    self.assertEqual(job_request['python_module'], 'trainer.task')
    self.assertEqual(job_request['package_uris'], ['gs://my_out_dir/staging/trainer.tar.gz'])

    self.assertTrue(find_key_value(cmd_list, '--job-dir', 'gs://my_out_dir'))
    self.assertTrue(find_key_endswith(cmd_list, '--analysis', 'my_analyze_dir'))
    self.assertTrue(find_startswith_endswith(cmd_list, '--train=', 'file*.csv'))
    self.assertIn('--transform', cmd_list)
    self.assertTrue(find_startswith_endswith(cmd_list, '--eval=', '*.tfrecords.gz'))
    self.assertTrue(find_key_value(cmd_list, '--key', 'value'))
