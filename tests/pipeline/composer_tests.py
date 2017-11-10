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

from google.datalab.contrib.pipeline.composer._composer import Composer


class TestCases(unittest.TestCase):

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.cloud.storage.Blob')
  @mock.patch('google.cloud.storage.Client')
  @mock.patch('google.datalab.contrib.pipeline.composer._api.Api.environment_details_get')
  def test_deploy(self, mock_environment_details, mock_client, mock_blob_class,
                  mock_default_context):
      # Happy path
      mock_environment_details.return_value = {
        'config': {
          'gcsDagLocation': 'gs://foo_bucket/dags'
        }
      }
      test_composer = Composer('foo_zone', 'foo_environment')
      test_composer.deploy('foo_name', 'foo_dag_string')
      mock_client.return_value.get_bucket.assert_called_with('foo_bucket')
      mock_blob_class.assert_called_with('dags/foo_name.py', mock.ANY)
      mock_blob = mock_blob_class.return_value
      mock_blob.upload_from_string.assert_called_with('foo_dag_string')

      # Only bucket with no path
      mock_environment_details.return_value = {
        'config': {
          'gcsDagLocation': 'gs://foo_bucket'
        }
      }
      test_composer = Composer('foo_zone', 'foo_environment')
      test_composer.deploy('foo_name', 'foo_dag_string')
      mock_client.return_value.get_bucket.assert_called_with('foo_bucket')
      mock_blob_class.assert_called_with('foo_name.py', mock.ANY)
      mock_blob = mock_blob_class.return_value
      mock_blob.upload_from_string.assert_called_with('foo_dag_string')

      # GCS dag location has additional parts
      mock_environment_details.return_value = {
        'config': {
          'gcsDagLocation': 'gs://foo_bucket/foo_random/dags'
        }
      }
      test_composer = Composer('foo_zone', 'foo_environment')
      test_composer.deploy('foo_name', 'foo_dag_string')
      mock_client.return_value.get_bucket.assert_called_with('foo_bucket')
      mock_blob_class.assert_called_with('foo_random/dags/foo_name.py', mock.ANY)
      mock_blob = mock_blob_class.return_value
      mock_blob.upload_from_string.assert_called_with('foo_dag_string')

  @mock.patch('google.datalab.contrib.pipeline.composer._api.Api.environment_details_get')
  def test_gcs_dag_location(self, mock_environment_details):
      # Composer returns good result
      mock_environment_details.return_value = {
        'config': {
          'gcsDagLocation': 'gs://foo_bucket/dags'
        }
      }
      test_composer = Composer('foo_zone', 'foo_environment')
      self.assertEqual('gs://foo_bucket/dags/', test_composer.gcs_dag_location)

      # Composer returns good result
      mock_environment_details.return_value = {
        'config': {
          'gcsDagLocation': 'gs://foo_bucket'
        }
      }
      test_composer = Composer('foo_zone', 'foo_environment')
      self.assertEqual('gs://foo_bucket/', test_composer.gcs_dag_location)

      # Composer returns empty result
      mock_environment_details.return_value = {}
      test_composer = Composer('foo_zone', 'foo_environment')
      with self.assertRaisesRegexp(
              ValueError, 'Dag location unavailable from Composer environment foo_environment'):
        test_composer.gcs_dag_location

      # Composer returns empty result
      mock_environment_details.return_value = {
        'config': {}
      }
      test_composer = Composer('foo_zone', 'foo_environment')
      with self.assertRaisesRegexp(
              ValueError, 'Dag location unavailable from Composer environment foo_environment'):
        test_composer.gcs_dag_location

      # Composer returns None result
      mock_environment_details.return_value = {
        'config': {
          'gcsDagLocation': None
        }
      }
      test_composer = Composer('foo_zone', 'foo_environment')
      with self.assertRaisesRegexp(
              ValueError,
              'Dag location None from Composer environment foo_environment is in incorrect format'):
        test_composer.gcs_dag_location

      # Composer returns incorrect formats
      mock_environment_details.return_value = {
        'config': {
          'gcsDagLocation': 'gs:/foo_bucket'
        }
      }
      test_composer = Composer('foo_zone', 'foo_environment')
      with self.assertRaisesRegexp(
              ValueError,
              ('Dag location gs:/foo_bucket from Composer environment foo_environment is in'
               ' incorrect format')):
        test_composer.gcs_dag_location

      mock_environment_details.return_value = {
        'config': {
          'gcsDagLocation': 'as://foo_bucket'
        }
      }
      test_composer = Composer('foo_zone', 'foo_environment')
      with self.assertRaisesRegexp(
              ValueError,
              ('Dag location as://foo_bucket from Composer environment foo_environment is in'
               ' incorrect format')):
        test_composer.gcs_dag_location
