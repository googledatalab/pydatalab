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

import google.datalab
import google.datalab.stackdriver.monitoring as gcm

DEFAULT_PROJECT = 'test'
PROJECT = 'my-project'
GROUP_IDS = ['GROUP-205', 'GROUP-101']
PARENT_IDS = [None, GROUP_IDS[0]]
DISPLAY_NAMES = ['All Instances', 'GCE Instances']
PARENT_DISPLAY_NAMES = ['', DISPLAY_NAMES[0]]
FILTER_STRINGS = ['resource.type = ends_with("instance")',
                  'resource.type = "gce_instance"']
IS_CLUSTERS = [False, True]


class TestCases(unittest.TestCase):

  def setUp(self):
    self.context = self._create_context(DEFAULT_PROJECT)
    self.groups = gcm.Groups(context=self.context)

  @mock.patch('google.datalab.Context.default')
  def test_constructor_minimal(self, mock_context_default):
    mock_context_default.return_value = self.context

    groups = gcm.Groups()

    self.assertIs(groups._context, self.context)
    self.assertIsNone(groups._group_dict)

    self.assertEqual(groups._client.project, DEFAULT_PROJECT)
    self.assertEqual(groups._client.connection.credentials,
                     self.context.credentials)

  def test_constructor_maximal(self):
    context = self._create_context(PROJECT)
    groups = gcm.Groups(context)
    self.assertIs(groups._context, context)
    self.assertIsNone(groups._group_dict)
    self.assertEqual(groups._client.project, PROJECT)
    self.assertEqual(groups._client.connection.credentials,
                     context.credentials)

  @mock.patch('google.cloud.monitoring.Client.list_groups')
  def test_list(self, mock_api_list_groups):
    mock_api_list_groups.return_value = self._list_groups_get_result(
        context=self.context)

    group_list = self.groups.list()

    mock_api_list_groups.assert_called_once_with()
    self.assertEqual(len(group_list), 2)
    self.assertEqual(group_list[0].id, GROUP_IDS[0])
    self.assertEqual(group_list[1].id, GROUP_IDS[1])

  @mock.patch('google.cloud.monitoring.Client.list_groups')
  def test_list_w_pattern_match(self, mock_api_list_groups):
    mock_api_list_groups.return_value = self._list_groups_get_result(
        context=self.context)

    group_list = self.groups.list(pattern='GCE*')

    mock_api_list_groups.assert_called_once_with()
    self.assertEqual(len(group_list), 1)
    self.assertEqual(group_list[0].id, GROUP_IDS[1])

  @mock.patch('google.cloud.monitoring.Client.list_groups')
  def test_list_caching(self, mock_gcloud_list_groups):
    mock_gcloud_list_groups.return_value = self._list_groups_get_result(
        context=self.context)

    actual_list1 = self.groups.list()
    actual_list2 = self.groups.list()

    mock_gcloud_list_groups.assert_called_once_with()
    self.assertEqual(actual_list1, actual_list2)

  @mock.patch('google.cloud.monitoring.Client.list_groups')
  def test_as_dataframe(self, mock_gcloud_list_groups):
    mock_gcloud_list_groups.return_value = self._list_groups_get_result(
        context=self.context)
    dataframe = self.groups.as_dataframe()
    mock_gcloud_list_groups.assert_called_once_with()

    expected_headers = list(gcm.Groups._DISPLAY_HEADERS)
    self.assertEqual(dataframe.columns.tolist(), expected_headers)
    self.assertEqual(dataframe.columns.names, [None])

    self.assertEqual(dataframe.index.tolist(), list(range(len(GROUP_IDS))))
    self.assertEqual(dataframe.index.names, [None])

    expected_values = [list(row) for row in
                       zip(GROUP_IDS, DISPLAY_NAMES, PARENT_IDS,
                           PARENT_DISPLAY_NAMES, IS_CLUSTERS, FILTER_STRINGS)]
    self.assertEqual(dataframe.values.tolist(), expected_values)

  @mock.patch('google.cloud.monitoring.Client.list_groups')
  def test_as_dataframe_w_all_args(self, mock_gcloud_list_groups):
    mock_gcloud_list_groups.return_value = self._list_groups_get_result(
        context=self.context)
    dataframe = self.groups.as_dataframe(pattern='*Instance*', max_rows=1)
    mock_gcloud_list_groups.assert_called_once_with()

    expected_headers = list(gcm.Groups._DISPLAY_HEADERS)
    self.assertEqual(dataframe.columns.tolist(), expected_headers)
    self.assertEqual(dataframe.index.tolist(), [0])
    self.assertEqual(dataframe.iloc[0, 0], GROUP_IDS[0])

  @staticmethod
  def _create_context(project_id):
    creds = AccessTokenCredentials('test_token', 'test_ua')
    return google.datalab.Context(project_id, creds)

  @staticmethod
  def _list_groups_get_result(context):
    client = gcm._utils.make_client(context=context)
    groups = []
    for group_id, parent_id, display_name, filter_string, is_cluster in \
            zip(GROUP_IDS, PARENT_IDS, DISPLAY_NAMES, FILTER_STRINGS, IS_CLUSTERS):
      group = client.group(group_id=group_id, display_name=display_name,
                           parent_id=parent_id, filter_string=filter_string,
                           is_cluster=is_cluster)
      groups.append(group)

    return groups
