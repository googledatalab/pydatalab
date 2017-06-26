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

import pandas

import google.datalab
import google.datalab.stackdriver.commands._monitoring as monitoring_commands

DEFAULT_PROJECT = 'test'
PROJECT = 'my-project'


class MockCredentials(Credentials):
    def __init__(self, token='token'):
        super(MockCredentials, self).__init__()
        self.token = token
        self.expiry = None

    def refresh(self, request):
        self.token += '1'


class TestCases(unittest.TestCase):
  def setUp(self):
    self.context = self._create_context(DEFAULT_PROJECT)

  @mock.patch('google.datalab.Context.default')
  def test_make_context(self, mock_context_default):
    mock_context_default.return_value = self.context
    new_context = monitoring_commands._make_context(PROJECT)
    self.assertEqual(new_context.project_id, PROJECT)
    self.assertEqual(new_context.credentials, self.context.credentials)

  @mock.patch('google.datalab.Context.default')
  def test_make_context_empty_project(self, mock_context_default):
    mock_context_default.return_value = self.context
    new_context = monitoring_commands._make_context('')
    self.assertEqual(new_context.project_id, DEFAULT_PROJECT)
    self.assertEqual(new_context.credentials, self.context.credentials)

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.stackdriver.commands._monitoring._render_dataframe')
  @mock.patch('google.datalab.stackdriver.monitoring.MetricDescriptors')
  def test_monitoring_metrics_list(self, mock_metric_descriptors, mock_render_dataframe,
                                   mock_context_default):
    METRIC_TYPES = ['compute.googleapis.com/instances/cpu/utilization',
                    'compute.googleapis.com/instances/cpu/usage_time']
    DATAFRAME = pandas.DataFrame(METRIC_TYPES, columns=['Metric type'])
    PATTERN = 'compute*cpu*'

    mock_context_default.return_value = self.context
    mock_metric_class = mock_metric_descriptors.return_value
    mock_metric_class.as_dataframe.return_value = DATAFRAME

    monitoring_commands._monitoring_metrics_list(
        {'project': PROJECT, 'type': PATTERN}, None)

    mock_metric_descriptors.assert_called_once()
    mock_metric_class.as_dataframe.assert_called_once_with(pattern=PATTERN)
    mock_render_dataframe.assert_called_once_with(DATAFRAME)

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.stackdriver.commands._monitoring._render_dataframe')
  @mock.patch('google.datalab.stackdriver.monitoring.ResourceDescriptors')
  def test_monitoring_resource_types_list(self, mock_resource_descriptors, mock_render_dataframe,
                                          mock_context_default):
    RESOURCE_TYPES = ['gce_instance', 'aws_ec2_instance']
    DATAFRAME = pandas.DataFrame(RESOURCE_TYPES, columns=['Resource type'])
    PATTERN = '*instance*'

    mock_context_default.return_value = self.context
    mock_resource_class = mock_resource_descriptors.return_value
    mock_resource_class.as_dataframe.return_value = DATAFRAME

    monitoring_commands._monitoring_resource_types_list(
        {'project': PROJECT, 'type': PATTERN}, None)

    mock_resource_descriptors.assert_called_once()
    mock_resource_class.as_dataframe.assert_called_once_with(pattern=PATTERN)
    mock_render_dataframe.assert_called_once_with(DATAFRAME)

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.stackdriver.commands._monitoring._render_dataframe')
  @mock.patch('google.datalab.stackdriver.monitoring.Groups')
  def test_monitoring_groups_list(self, mock_groups, mock_render_dataframe,
                                  mock_context_default):
    GROUP_IDS = ['GROUP-205', 'GROUP-101']
    DATAFRAME = pandas.DataFrame(GROUP_IDS, columns=['Group ID'])
    PATTERN = 'GROUP-*'

    mock_context_default.return_value = self.context
    mock_group_class = mock_groups.return_value
    mock_group_class.as_dataframe.return_value = DATAFRAME

    monitoring_commands._monitoring_groups_list(
        {'project': PROJECT, 'name': PATTERN}, None)

    mock_groups.assert_called_once()
    mock_group_class.as_dataframe.assert_called_once_with(pattern=PATTERN)
    mock_render_dataframe.assert_called_once_with(DATAFRAME)

  @staticmethod
  def _create_context(project_id):
    creds = MockCredentials()
    return google.datalab.Context(project_id, creds)
