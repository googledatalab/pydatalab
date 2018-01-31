# Copyright 2015 Google Inc. All rights reserved.
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

import google
import google.auth
import unittest
import mock

from google.datalab.contrib.pipeline.airflow._airflow import Airflow


class TestCases(unittest.TestCase):

  @staticmethod
  def _create_context():
      project_id = 'test'
      creds = mock.Mock(spec=google.auth.credentials.Credentials)
      return google.datalab.Context(project_id, creds)

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.storage.Bucket')
  def test_deploy(self, mock_bucket_class, mock_default_context):
      context = TestCases._create_context()
      mock_default_context.return_value = context

      # Happy path
      test_airflow = Airflow('foo_bucket', 'foo_path')
      test_airflow.deploy('foo_name', 'foo_dag_string')
      mock_bucket_class.assert_called_with('foo_bucket')
      mock_bucket_class.return_value.object.assert_called_with('foo_path/foo_name.py')
      mock_bucket_class.return_value.object.return_value.write_stream.assert_called_with(
          'foo_dag_string', 'text/plain')

      # Only bucket
      test_airflow = Airflow('foo_bucket')
      test_airflow.deploy('foo_name', 'foo_dag_string')
      mock_bucket_class.assert_called_with('foo_bucket')
      mock_bucket_class.return_value.object.assert_called_with('foo_name.py')
      mock_bucket_class.return_value.object.return_value.write_stream.assert_called_with(
          'foo_dag_string', 'text/plain')

      # Only bucket with path as '/'
      test_airflow = Airflow('foo_bucket', '/')
      test_airflow.deploy('foo_name', 'foo_dag_string')
      mock_bucket_class.assert_called_with('foo_bucket')
      mock_bucket_class.return_value.object.assert_called_with('/foo_name.py')
      mock_bucket_class.return_value.object.return_value.write_stream.assert_called_with(
          'foo_dag_string', 'text/plain')

      # GCS dag location has additional parts
      test_airflow = Airflow('foo_bucket', 'foo_path1/foo_path2')
      test_airflow.deploy('foo_name', 'foo_dag_string')
      mock_bucket_class.assert_called_with('foo_bucket')
      mock_bucket_class.return_value.object.assert_called_with('foo_path1/foo_path2/foo_name.py')
      mock_bucket_class.return_value.object.return_value.write_stream.assert_called_with(
          'foo_dag_string', 'text/plain')
