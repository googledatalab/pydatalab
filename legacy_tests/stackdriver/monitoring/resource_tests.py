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
import unittest

import google.cloud.monitoring

import google.auth
import datalab.context
import datalab.stackdriver.monitoring as gcm

PROJECT = 'my-project'
RESOURCE_TYPES = ['gce_instance', 'aws_ec2_instance']
DISPLAY_NAMES = ['GCE VM Instance', 'Amazon EC2 Instance']

LABELS = [dict(key='instance_id', value_type='STRING',
               description='VM instance ID'),
          dict(key='project_id', value_type='STRING',
               description='Project ID')]
FILTER_STRING = 'resource.type = ends_with("instance")'


class TestCases(unittest.TestCase):

  def setUp(self):
    self.context = self._create_context()
    self.descriptors = gcm.ResourceDescriptors(context=self.context)

  @mock.patch('datalab.context._context.Context.default')
  def test_constructor_minimal(self, mock_context_default):
    mock_context_default.return_value = self.context

    descriptors = gcm.ResourceDescriptors()

    expected_client = gcm._utils.make_client(context=self.context)
    self.assertEqual(descriptors._client.project, expected_client.project)
    self.assertEqual(descriptors._client._connection.credentials,
                     expected_client._connection.credentials)

    self.assertIsNone(descriptors._filter_string)
    self.assertIsNone(descriptors._descriptors)

  def test_constructor_maximal(self):
    context = self._create_context(PROJECT)
    descriptors = gcm.ResourceDescriptors(
        filter_string=FILTER_STRING, project_id=PROJECT, context=context)

    expected_client = gcm._utils.make_client(
        context=context, project_id=PROJECT)
    self.assertEqual(descriptors._client.project, expected_client.project)
    self.assertEqual(descriptors._client._connection.credentials,
                     expected_client._connection.credentials)

    self.assertEqual(descriptors._filter_string, FILTER_STRING)
    self.assertIsNone(descriptors._descriptors)

  @mock.patch('google.cloud.monitoring.Client.list_resource_descriptors')
  def test_list(self, mock_api_list_descriptors):
    mock_api_list_descriptors.return_value = self._list_resources_get_result()

    resource_descriptor_list = self.descriptors.list()

    mock_api_list_descriptors.assert_called_once_with(filter_string=None)
    self.assertEqual(len(resource_descriptor_list), 2)
    self.assertEqual(resource_descriptor_list[0].type, RESOURCE_TYPES[0])
    self.assertEqual(resource_descriptor_list[1].type, RESOURCE_TYPES[1])

  @mock.patch('google.cloud.monitoring.Client.list_resource_descriptors')
  def test_list_w_api_filter(self, mock_api_list_descriptors):
    mock_api_list_descriptors.return_value = self._list_resources_get_result()

    descriptors = gcm.ResourceDescriptors(
        filter_string=FILTER_STRING, context=self.context)
    resource_descriptor_list = descriptors.list()

    mock_api_list_descriptors.assert_called_once_with(
        filter_string=FILTER_STRING)
    self.assertEqual(len(resource_descriptor_list), 2)
    self.assertEqual(resource_descriptor_list[0].type, RESOURCE_TYPES[0])
    self.assertEqual(resource_descriptor_list[1].type, RESOURCE_TYPES[1])

  @mock.patch('google.cloud.monitoring.Client.list_resource_descriptors')
  def test_list_w_pattern_match(self, mock_api_list_descriptors):
    mock_api_list_descriptors.return_value = self._list_resources_get_result()

    resource_descriptor_list = self.descriptors.list(pattern='*ec2*')

    mock_api_list_descriptors.assert_called_once_with(filter_string=None)
    self.assertEqual(len(resource_descriptor_list), 1)
    self.assertEqual(resource_descriptor_list[0].type, RESOURCE_TYPES[1])

  @mock.patch('google.cloud.monitoring.Client.list_resource_descriptors')
  def test_list_caching(self, mock_gcloud_list_descriptors):
    mock_gcloud_list_descriptors.return_value = (
        self._list_resources_get_result())

    actual_list1 = self.descriptors.list()
    actual_list2 = self.descriptors.list()

    mock_gcloud_list_descriptors.assert_called_once_with(filter_string=None)
    self.assertEqual(actual_list1, actual_list2)

  @mock.patch('datalab.stackdriver.monitoring.ResourceDescriptors.list')
  def test_as_dataframe(self, mock_datalab_list_descriptors):
    mock_datalab_list_descriptors.return_value = (
        self._list_resources_get_result())
    dataframe = self.descriptors.as_dataframe()
    mock_datalab_list_descriptors.assert_called_once_with('*')

    expected_headers = list(gcm.ResourceDescriptors._DISPLAY_HEADERS)
    self.assertEqual(dataframe.columns.tolist(), expected_headers)
    self.assertEqual(dataframe.columns.names, [None])

    self.assertEqual(dataframe.index.tolist(), list(range(len(RESOURCE_TYPES))))
    self.assertEqual(dataframe.index.names, [None])

    expected_labels = 'instance_id, project_id'
    expected_values = [
        [resource_type, display_name, expected_labels]
        for resource_type, display_name in zip(RESOURCE_TYPES, DISPLAY_NAMES)]
    self.assertEqual(dataframe.values.tolist(), expected_values)

  @mock.patch('datalab.stackdriver.monitoring.ResourceDescriptors.list')
  def test_as_dataframe_w_all_args(self, mock_datalab_list_descriptors):
    mock_datalab_list_descriptors.return_value = (
        self._list_resources_get_result())
    dataframe = self.descriptors.as_dataframe(pattern='*instance*', max_rows=1)
    mock_datalab_list_descriptors.assert_called_once_with('*instance*')

    expected_headers = list(gcm.ResourceDescriptors._DISPLAY_HEADERS)
    self.assertEqual(dataframe.columns.tolist(), expected_headers)
    self.assertEqual(dataframe.index.tolist(), [0])
    self.assertEqual(dataframe.iloc[0, 0], RESOURCE_TYPES[0])

  @staticmethod
  def _create_context(project_id='test'):
    creds = mock.Mock(spec=google.auth.credentials.Credentials)
    return datalab.context.Context(project_id, creds)

  @staticmethod
  def _list_resources_get_result():
    all_labels = [google.cloud.monitoring.LabelDescriptor(**labels)
                  for labels in LABELS]
    descriptors = [
        google.cloud.monitoring.ResourceDescriptor(
            name=None, type_=resource_type, display_name=display_name,
            description=None, labels=all_labels,
        )
        for resource_type, display_name in zip(RESOURCE_TYPES, DISPLAY_NAMES)]
    return descriptors
