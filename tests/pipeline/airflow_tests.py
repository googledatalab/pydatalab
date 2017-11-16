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

import unittest
import mock

from google.datalab.contrib.pipeline.airflow._airflow import Airflow


class TestCases(unittest.TestCase):

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.cloud.storage.Blob')
  @mock.patch('google.cloud.storage.Client')
  def test_deploy(self, mock_client, mock_blob_class, mock_default_context):
      # Happy path
      test_airflow = Airflow('foo_bucket', 'foo_path')
      test_airflow.deploy('foo_name', 'foo_dag_string')
      mock_client.return_value.get_bucket.assert_called_with('foo_bucket')
      mock_blob_class.assert_called_with('foo_path/foo_name.py', mock.ANY)
      mock_blob = mock_blob_class.return_value
      mock_blob.upload_from_string.assert_called_with('foo_dag_string')

      # Only bucket
      test_airflow = Airflow('foo_bucket')
      test_airflow.deploy('foo_name', 'foo_dag_string')
      mock_client.return_value.get_bucket.assert_called_with('foo_bucket')
      mock_blob_class.assert_called_with('foo_name.py', mock.ANY)
      mock_blob = mock_blob_class.return_value
      mock_blob.upload_from_string.assert_called_with('foo_dag_string')

      # Only bucket with path as '/'
      test_airflow = Airflow('foo_bucket', '/')
      test_airflow.deploy('foo_name', 'foo_dag_string')
      mock_client.return_value.get_bucket.assert_called_with('foo_bucket')
      mock_blob_class.assert_called_with('/foo_name.py', mock.ANY)
      mock_blob = mock_blob_class.return_value
      mock_blob.upload_from_string.assert_called_with('foo_dag_string')

      # GCS dag location has additional parts
      test_airflow = Airflow('foo_bucket', 'foo_path1/foo_path2')
      test_airflow.deploy('foo_name', 'foo_dag_string')
      mock_client.return_value.get_bucket.assert_called_with('foo_bucket')
      mock_blob_class.assert_called_with('foo_path1/foo_path2/foo_name.py', mock.ANY)
      mock_blob = mock_blob_class.return_value
      mock_blob.upload_from_string.assert_called_with('foo_dag_string')
