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

import google.auth
import google.datalab.utils
from google.datalab.contrib.pipeline.composer._composer import Composer


class TestCases(unittest.TestCase):

  @mock.patch('google.cloud.storage.Client')
  @mock.patch('google.cloud.storage.Blob')
  @mock.patch('google.cloud.storage.Client.get_bucket')
  def test_deploy(self, mock_client_get_bucket, mock_blob_class, mock_client):
      mock_client_get_bucket.return_value = mock.Mock(spec=google.cloud.storage.Bucket)
      mock_blob = mock_blob_class.return_value
      test_composer = Composer('foo_zone', 'foo_environment')
      test_composer.deploy('foo_name', 'foo_dag_string')
      mock_blob.upload_from_string.assert_called_with('foo_dag_string')
