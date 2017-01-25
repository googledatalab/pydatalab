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
from oauth2client.client import AccessTokenCredentials
import unittest

import google.datalab.context
import google.datalab.stackdriver.monitoring as gcm


class TestCases(unittest.TestCase):

  def test_make_client(self):
    project_id = 'project_id'
    context = self._create_context()
    client = gcm._utils.make_client(project_id, context)

    self.assertEqual(client.project, project_id)
    self.assertEqual(client.connection.credentials, context.credentials)
    self.assertEqual(client._connection_class.USER_AGENT, 'pydatalab/v0')

  @mock.patch('google.datalab.context._context.Context.default')
  def test_make_client_w_defaults(self, mock_context_default):
    default_context = self._create_context()
    mock_context_default.return_value = default_context
    client = gcm._utils.make_client()

    self.assertEqual(client.project, default_context.project_id)
    self.assertEqual(
        client.connection.credentials, default_context.credentials)
    self.assertEqual(client._connection_class.USER_AGENT, 'pydatalab/v0')

  @staticmethod
  def _create_context(project_id='test'):
    creds = AccessTokenCredentials('test_token', 'test_ua')
    return google.datalab.context.Context(project_id, creds)
