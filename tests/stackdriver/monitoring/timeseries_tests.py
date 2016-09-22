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
import datetime
import mock
from oauth2client.client import AccessTokenCredentials
import unittest

from google.cloud.monitoring import Resource
from google.cloud.monitoring import Metric
from google.cloud.monitoring import Query as BaseQuery
from google.cloud.monitoring import TimeSeries

import datalab.context
import datalab.stackdriver.monitoring as gcm


PROJECT = 'my-project'

METRIC_TYPE = 'compute.googleapis.com/instance/cpu/utilization'
RESOURCE_TYPE = 'gce_instance'
INSTANCE_NAMES = ['instance-1', 'instance-2']
INSTANCE_ZONES = ['us-east1-a', 'us-east1-b']
INSTANCE_IDS = ['1234567890123456789', '9876543210987654321']


class TestCases(unittest.TestCase):

  @mock.patch('datalab.context._context.Context.default')
  def test_constructor_minimal(self, mock_context_default):
    default_context = self._create_context()
    mock_context_default.return_value = default_context

    query = gcm.Query()

    expected_client = gcm._utils.make_client(context=default_context)
    self.assertEqual(query._client.project, expected_client.project)
    self.assertEqual(query._client.connection.credentials,
                     expected_client.connection.credentials)

    self.assertEqual(query._filter.metric_type, BaseQuery.DEFAULT_METRIC_TYPE)

    self.assertIsNone(query._start_time)
    self.assertIsNone(query._end_time)

    self.assertIsNone(query._per_series_aligner)
    self.assertIsNone(query._alignment_period_seconds)
    self.assertIsNone(query._cross_series_reducer)
    self.assertEqual(query._group_by_fields, ())

  def test_constructor_maximal(self):
    UPTIME_METRIC = 'compute.googleapis.com/instance/uptime'
    T1 = datetime.datetime(2016, 4, 7, 2, 30, 30)
    DAYS, HOURS, MINUTES = 1, 2, 3
    T0 = T1 - datetime.timedelta(days=DAYS, hours=HOURS, minutes=MINUTES)

    context = self._create_context(PROJECT)
    query = gcm.Query(UPTIME_METRIC,
                      end_time=T1, days=DAYS, hours=HOURS, minutes=MINUTES,
                      project_id=PROJECT, context=context)

    expected_client = gcm._utils.make_client(
        context=context, project_id=PROJECT)
    self.assertEqual(query._client.project, expected_client.project)
    self.assertEqual(query._client.connection.credentials,
                     expected_client.connection.credentials)

    self.assertEqual(query._filter.metric_type, UPTIME_METRIC)

    self.assertEqual(query._start_time, T0)
    self.assertEqual(query._end_time, T1)

    self.assertIsNone(query._per_series_aligner)
    self.assertIsNone(query._alignment_period_seconds)
    self.assertIsNone(query._cross_series_reducer)
    self.assertEqual(query._group_by_fields, ())

  @mock.patch('datalab.stackdriver.monitoring.Query.iter')
  def test_labels_as_dataframe(self, mock_query_iter):
    mock_query_iter.return_value = self._query_iter_get_result()
    query = gcm.Query(context=self._create_context(PROJECT))
    dataframe = query.labels_as_dataframe()

    mock_query_iter.assert_called_once_with(headers_only=True)
    NUM_INSTANCES = len(INSTANCE_IDS)

    self.assertEqual(dataframe.shape, (NUM_INSTANCES, 5))

    expected_values = [
        [RESOURCE_TYPE, PROJECT, zone, instance_id, instance_name]
        for zone, instance_id, instance_name
        in zip(INSTANCE_ZONES, INSTANCE_IDS, INSTANCE_NAMES)]
    self.assertEqual(dataframe.values.tolist(), expected_values)

    expected_headers = [
        ('resource', 'type'),
        ('resource.labels', 'project_id'),
        ('resource.labels', 'zone'),
        ('resource.labels', 'instance_id'),
        ('metric.labels', 'instance_name')
    ]
    self.assertEqual(dataframe.columns.tolist(), expected_headers)
    self.assertEqual(dataframe.columns.names, [None, None])

    self.assertEqual(dataframe.index.tolist(), list(range(NUM_INSTANCES)))
    self.assertEqual(dataframe.index.names, [None])

  @mock.patch('datalab.stackdriver.monitoring.Query.iter')
  def test_labels_as_dataframe_w_no_data(self, mock_query_iter):
    mock_query_iter.return_value = []
    query = gcm.Query(context=self._create_context())
    dataframe = query.labels_as_dataframe()

    mock_query_iter.assert_called_once_with(headers_only=True)
    self.assertEqual(dataframe.shape, (0, 0))
    self.assertIsNone(dataframe.columns.name)
    self.assertIsNone(dataframe.index.name)

  @staticmethod
  def _create_context(project_id='test'):
    creds = AccessTokenCredentials('test_token', 'test_ua')
    return datalab.context.Context(project_id, creds)

  @staticmethod
  def _query_iter_get_result():
    METRIC_LABELS = list({'instance_name': name} for name in INSTANCE_NAMES)
    RESOURCE_LABELS = list({
        'project_id': PROJECT,
        'zone': zone,
        'instance_id': instance_id,
    } for zone, instance_id in zip(INSTANCE_ZONES, INSTANCE_IDS))

    for metric_labels, resource_labels in zip(METRIC_LABELS, RESOURCE_LABELS):
      yield TimeSeries(
        metric=Metric(type=METRIC_TYPE, labels=metric_labels),
        resource=Resource(type=RESOURCE_TYPE, labels=resource_labels),
        metric_kind='GAUGE',
        value_type='DOUBLE',
        points=[],
      )
