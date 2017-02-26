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

import pandas

import datalab.stackdriver.commands._monitoring as monitoring_commands

PROJECT = 'my-project'


class TestCases(unittest.TestCase):

  @mock.patch('datalab.stackdriver.commands._monitoring._render_dataframe')
  @mock.patch('datalab.stackdriver.monitoring.MetricDescriptors')
  def test_list_metric_descriptors(self, mock_metric_descriptors, mock_render_dataframe):
    METRIC_TYPES = ['compute.googleapis.com/instances/cpu/utilization',
                    'compute.googleapis.com/instances/cpu/usage_time']
    DATAFRAME = pandas.DataFrame(METRIC_TYPES, columns=['Metric type'])
    PATTERN = 'compute*cpu*'

    mock_metric_class = mock_metric_descriptors.return_value
    mock_metric_class.as_dataframe.return_value = DATAFRAME

    monitoring_commands._list_metric_descriptors(
        {'project': PROJECT, 'type': PATTERN}, None)

    mock_metric_descriptors.assert_called_once_with(project_id=PROJECT)
    mock_metric_class.as_dataframe.assert_called_once_with(pattern=PATTERN)
    mock_render_dataframe.assert_called_once_with(DATAFRAME)

  @mock.patch('datalab.stackdriver.commands._monitoring._render_dataframe')
  @mock.patch('datalab.stackdriver.monitoring.ResourceDescriptors')
  def test_list_resource_descriptors(self, mock_resource_descriptors, mock_render_dataframe):
    RESOURCE_TYPES = ['gce_instance', 'aws_ec2_instance']
    DATAFRAME = pandas.DataFrame(RESOURCE_TYPES, columns=['Resource type'])
    PATTERN = '*instance*'

    mock_resource_class = mock_resource_descriptors.return_value
    mock_resource_class.as_dataframe.return_value = DATAFRAME

    monitoring_commands._list_resource_descriptors(
        {'project': PROJECT, 'type': PATTERN}, None)

    mock_resource_descriptors.assert_called_once_with(project_id=PROJECT)
    mock_resource_class.as_dataframe.assert_called_once_with(pattern=PATTERN)
    mock_render_dataframe.assert_called_once_with(DATAFRAME)

  @mock.patch('datalab.stackdriver.commands._monitoring._render_dataframe')
  @mock.patch('datalab.stackdriver.monitoring.Groups')
  def test_list_groups(self, mock_groups, mock_render_dataframe):
    GROUP_IDS = ['GROUP-205', 'GROUP-101']
    DATAFRAME = pandas.DataFrame(GROUP_IDS, columns=['Group ID'])
    PATTERN = 'GROUP-*'

    mock_group_class = mock_groups.return_value
    mock_group_class.as_dataframe.return_value = DATAFRAME

    monitoring_commands._list_groups(
        {'project': PROJECT, 'name': PATTERN}, None)

    mock_groups.assert_called_once_with(project_id=PROJECT)
    mock_group_class.as_dataframe.assert_called_once_with(pattern=PATTERN)
    mock_render_dataframe.assert_called_once_with(DATAFRAME)
