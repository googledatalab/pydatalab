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
from google.datalab.contrib.pipeline.composer._api import Api


class TestCases(unittest.TestCase):

  TEST_PROJECT_ID = 'test_project'

  def validate(self, mock_http_request, expected_url, expected_args=None, expected_data=None,
               expected_headers=None, expected_method=None):
    url = mock_http_request.call_args[0][0]
    kwargs = mock_http_request.call_args[1]
    self.assertEquals(expected_url, url)
    if expected_args is not None:
      self.assertEquals(expected_args, kwargs['args'])
    else:
      self.assertNotIn('args', kwargs)
    if expected_data is not None:
      self.assertEquals(expected_data, kwargs['data'])
    else:
      self.assertNotIn('data', kwargs)
    if expected_headers is not None:
      self.assertEquals(expected_headers, kwargs['headers'])
    else:
      self.assertNotIn('headers', kwargs)
    if expected_method is not None:
      self.assertEquals(expected_method, kwargs['method'])
    else:
      self.assertNotIn('method', kwargs)

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.utils.Http.request')
  def test_environment_details_get(self, mock_http_request, mock_context_default):
    mock_context_default.return_value = TestCases._create_context()
    Api.environment_details_get('ZONE', 'ENVIRONMENT')
    self.validate(mock_http_request,
                  'https://composer.googleapis.com/v1alpha1/projects/test_project/locations/ZONE/'
                  'environments/ENVIRONMENT')

  @staticmethod
  def _create_context():
    project_id = TestCases.TEST_PROJECT_ID
    creds = mock.Mock(spec=google.auth.credentials.Credentials)
    return google.datalab.Context(project_id, creds)
