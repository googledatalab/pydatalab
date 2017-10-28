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

from __future__ import absolute_import
from __future__ import unicode_literals
import unittest
import mock

from google.datalab import Context
from google.datalab.utils import _utils as du


class TestCases(unittest.TestCase):

  def test_credentials(self):
    dummy_creds = {}
    c = Context('test_project', credentials=dummy_creds)

    self.assertEqual(c.credentials, dummy_creds)

    dummy_creds = {'test': 'test'}
    c.set_credentials(dummy_creds)
    self.assertEqual(c.credentials, dummy_creds)

  def test_config(self):
    dummy_config = {}
    c = Context('test_project', credentials=None, config=dummy_config)

    self.assertEqual(c.config, dummy_config)

    dummy_config = {'test': 'test'}
    c.set_config(dummy_config)
    self.assertEqual(c.config, dummy_config)

    c = Context('test_project', None, None)
    self.assertEqual(c.config, Context._get_default_config())

  def test_project(self):
    dummy_project = 'test_project'
    c = Context(dummy_project, credentials=None, config=None)

    self.assertEqual(c.project_id, dummy_project)

    dummy_project = 'test_project2'
    c.set_project_id(dummy_project)
    self.assertEqual(c.project_id, dummy_project)

    c = Context(None, None, None)
    with self.assertRaises(Exception):
      print(c.project_id)

  @mock.patch('google.datalab.utils._utils.get_credentials')
  @mock.patch('google.datalab.utils._utils.get_default_project_id')
  @mock.patch('google.datalab.utils._utils.save_project_id')
  def test_default_project(self, mock_save_project_id, mock_get_default_project_id,
                           mock_get_credentials):
    # verify setting the project on a default Context object sets
    # the global default project

    global default_project
    default_project = ''

    def save_project(project=None):
      global default_project
      default_project = project

    def get_project():
      global default_project
      return default_project

    mock_save_project_id.side_effect = save_project
    mock_get_default_project_id.side_effect = get_project
    mock_get_credentials.return_value = ''

    c = Context.default()
    dummy_project = 'test_project3'
    c.set_project_id(dummy_project)
    self.assertEqual(du.get_default_project_id(), dummy_project)

  @mock.patch('google.datalab.utils._utils.get_credentials')
  def test_is_signed_in(self, mock_get_credentials):
    mock_get_credentials.side_effect = Exception('No creds!')
    self.assertFalse(Context._is_signed_in())

    mock_get_credentials.return_value = {}
    mock_get_credentials.side_effect = None
    self.assertTrue(Context._is_signed_in())

  @mock.patch('google.datalab.utils._utils.get_credentials')
  @mock.patch('google.datalab.utils._utils.get_default_project_id')
  @mock.patch('google.datalab.utils._utils.save_project_id')
  def test_default_context(self, mock_save_project_id, mock_get_default_project_id,
                           mock_get_credentials):

    mock_get_default_project_id.return_value = 'default_project'
    mock_get_credentials.return_value = ''

    c = Context.default()
    default_project = c.project_id
    self.assertEqual(default_project, 'default_project')

    # deliberately change the default project and make sure it's reset
    c.set_project_id('test_project4')

    self.assertEqual(Context.default().project_id, 'default_project')
