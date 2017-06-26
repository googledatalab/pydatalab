# Copyright 2016 Google Inc. All rights reserved.
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

from __future__ import absolute_import
import mock
from google.auth.credentials import Credentials
import unittest

import google.datalab
import google.datalab.stackdriver.monitoring as gcm


class MockCredentials(Credentials):
    def __init__(self, token='token'):
        super(MockCredentials, self).__init__()
        self.token = token
        self.expiry = None

    def refresh(self, request):
        self.token += '1'


class TestCases(unittest.TestCase):

  def test_make_client(self):
    context = self._create_context()
    client = gcm._utils.make_client(context)

    self.assertEqual(client.project, context.project_id)
    self.assertEqual(client._connection.credentials, context.credentials)
    self.assertEqual(client.user_agent, 'pydatalab/v0')

  @mock.patch('google.datalab.Context.default')
  def test_make_client_w_defaults(self, mock_context_default):
    default_context = self._create_context()
    mock_context_default.return_value = default_context
    client = gcm._utils.make_client()

    self.assertEqual(client.project, default_context.project_id)
    self.assertEqual(
        client._connection.credentials, default_context.credentials)
    self.assertEqual(client.user_agent, 'pydatalab/v0')

  @staticmethod
  def _create_context():
    creds = MockCredentials()
    return google.datalab.Context('test_project', creds)
