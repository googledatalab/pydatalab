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

from __future__ import absolute_import
from __future__ import unicode_literals
import imp
import unittest
import pytz
import mock
import os

import google.datalab.utils._utils as _utils
import google.datalab.utils._iterator as _iterator
from datetime import datetime
import google.auth
import google.auth.exceptions


class TestCases(unittest.TestCase):

  @staticmethod
  def _get_data():
    m = imp.new_module('baz')
    exec('x = 99', m.__dict__)
    data = {
      'foo': {
        'bar': {
          'xyz': 0
        },
        'm': m
      }
    }
    return data

  def test_no_entry(self):
    data = TestCases._get_data()
    self.assertIsNone(_utils.get_item(data, ''))
    self.assertIsNone(_utils.get_item(data, 'x'))
    self.assertIsNone(_utils.get_item(data, 'bar.x'))
    self.assertIsNone(_utils.get_item(data, 'foo.bar.x'))
    self.assertIsNone(_utils.get_item(globals(), 'datetime.bar.x'))

  def test_entry(self):
    data = TestCases._get_data()
    self.assertEquals(data['foo']['bar']['xyz'], _utils.get_item(data, 'foo.bar.xyz'))
    self.assertEquals(data['foo']['bar'], _utils.get_item(data, 'foo.bar'))
    self.assertEquals(data['foo'], _utils.get_item(data, 'foo'))
    self.assertEquals(data['foo']['m'], _utils.get_item(data, 'foo.m'))
    self.assertEquals(99, _utils.get_item(data, 'foo.m.x'))

  def test_compare_datetimes(self):
    t1, t2 = datetime(2017, 2, 2, 12, 0, 0), datetime(2017, 2, 2, 12, 0, 0)
    self.assertEquals(_utils.compare_datetimes(t1, t2), 0)

    t2 = t2.replace(hour=11)
    self.assertEquals(_utils.compare_datetimes(t1, t2), 1)

  def test_compare_datetimes_tz(self):
    t1 = datetime(2017, 2, 2, 12, 0, 0)
    t2 = datetime(2017, 2, 2, 12, 0, 0, tzinfo=pytz.timezone('US/Eastern'))

    self.assertEquals(_utils.compare_datetimes(t1, t2), -1)

    t1 = t1.replace(tzinfo=pytz.timezone('US/Pacific'))

    self.assertEquals(_utils.compare_datetimes(t1, t2), 1)

  @mock.patch('os.path.expanduser')
  def test_get_config_dir(self, mock_expand_user):
    mock_expand_user.return_value = 'user/relative/path'
    with mock.patch.dict(os.environ, {'CLOUDSDK_CONFIG': 'test/path'}):
      self.assertEquals(_utils.get_config_dir(), 'test/path')

    self.assertEquals(_utils.get_config_dir(), 'user/relative/path/.config/gcloud')

  @mock.patch('os.name', 'nt')
  @mock.patch('os.path.join')
  def test_get_config_dir_win(self, mock_path_join):
    mock_path_join.side_effect = lambda x, y: x + y
    self.assertEquals(_utils.get_config_dir(), 'C:\\gcloud')

    mock_path_join.side_effect = lambda x, y: x + '\\' + y
    with mock.patch.dict(os.environ, {'APPDATA': 'test\\path'}):
      self.assertEquals(_utils.get_config_dir(), 'test\\path\\gcloud')

  @mock.patch('google.datalab.utils._utils._in_datalab_docker')
  @mock.patch('google.auth.credentials.with_scopes_if_required')
  @mock.patch('google.auth.default')
  @mock.patch('os.path.exists')
  def test_get_credentials_from_file(self, mock_path_exists, mock_google_auth_default,
                                     mock_with_scopes_if_required, mock_in_datalab):
    # If application default credentials exist, use them
    creds = mock.Mock(spec=google.auth.credentials.Credentials)
    mock_google_auth_default.return_value = [creds, '']
    _utils.get_credentials()
    mock_google_auth_default.assert_called_once()
    mock_with_scopes_if_required.assert_called_once()

    # If application default credentials are not defined, should load from file
    test_creds = '''
      {
        "data": [{
          "key": {
            "type": "google-cloud-sdk"
          },
          "credential": {
            "access_token": "test-access-token",
            "client_id": "test-id",
            "client_secret": "test-secret",
            "refresh_token": "test-token",
            "token_expiry": "test-expiry",
            "token_uri": "test-url",
            "user_agent": "test-agent",
            "invalid": "false"
          }
        }]
      }
    '''
    with mock.patch('google.datalab.utils._utils.open', mock.mock_open(read_data=test_creds)):
      mock_google_auth_default.side_effect = Exception
      cred = _utils.get_credentials()

      self.assertEquals(cred.token, 'test-access-token')

    mock_path_exists.return_value = False
    with self.assertRaises(Exception):
      cred = _utils.get_credentials()

    # If default creds are not defined, and no file exists with credentials, throw
    # something more meaningful.
    mock_google_auth_default.side_effect = google.auth.exceptions.DefaultCredentialsError
    with self.assertRaisesRegexp(Exception,
                                 'No application credentials found. Perhaps you should sign in'):
      cred = _utils.get_credentials()

  @mock.patch('subprocess.call')
  @mock.patch('os.path.exists')
  def test_save_project_id(self, mock_path_exists, mock_subprocess_call):
    _utils.save_project_id('test-project')
    mock_subprocess_call.assert_called_with([
      'gcloud', 'config', 'set', 'project', 'test-project'
    ])

    mock_subprocess_call.side_effect = Exception

    test_config = '''
      {
        "project_id": ""
      }
    '''
    opener = mock.mock_open(read_data=test_config)
    with mock.patch('google.datalab.utils._utils.open', opener):
      _utils.save_project_id('test-project')
      opener.assert_has_calls([mock.call().write('{"project_id": "test-project"}')])

  @mock.patch('subprocess.Popen')
  @mock.patch('os.path.exists')
  def test_get_default_project_id(self, mock_path_exists, mock_subprocess_call):
    mock_subprocess_call.return_value.communicate.return_value = ('test-project', '')
    mock_subprocess_call.return_value.poll.return_value = 0
    self.assertEquals(_utils.get_default_project_id(), 'test-project')
    mock_subprocess_call.assert_called_with(
      ['gcloud', 'config', 'list', '--format', 'value(core.project)'], stdout=-1)

    mock_subprocess_call.side_effect = Exception

    test_config = '''
      {
        "project_id": "test-project2"
      }
    '''
    opener = mock.mock_open(read_data=test_config)
    with mock.patch('google.datalab.utils._utils.open', opener):
      self.assertEquals(_utils.get_default_project_id(), 'test-project2')

    mock_path_exists.return_value = False
    self.assertIsNone(_utils.get_default_project_id())

    with mock.patch.dict(os.environ, {'PROJECT_ID': 'test-project3'}):
      self.assertEquals(_utils.get_default_project_id(), 'test-project3')

  def test_iterator(self):
    max_count = 100
    page_size = 10

    def limited_retriever(next_item, running_count):
      next_item = next_item or 1
      result_count = min(page_size, max_count - running_count)
      if result_count <= 0:
        return [], None
      return range(next_item, next_item + result_count), next_item + result_count

    read_count = 0
    for item in _iterator.Iterator(limited_retriever):
      read_count += 1
      self.assertLessEqual(read_count, max_count)
