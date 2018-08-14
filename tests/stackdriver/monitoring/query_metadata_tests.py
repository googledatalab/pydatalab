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

from google.cloud.logging.resource import Resource
from google.cloud.monitoring_v3.types import Metric
from google.cloud.monitoring_v3.types import TimeSeries

import google.auth
import google.datalab
import google.datalab.stackdriver.monitoring as gcm


PROJECT = 'my-project'

METRIC_TYPE = 'compute.googleapis.com/instance/cpu/utilization'
RESOURCE_TYPE = 'gce_instance'
INSTANCE_NAMES = ['instance-1', 'instance-2']
INSTANCE_ZONES = ['us-east1-a', 'us-east1-b']
INSTANCE_IDS = ['1234567890123456789', '9876543210987654321']


class TestCases(unittest.TestCase):

  def setUp(self):
    creds = mock.Mock(spec=google.auth.credentials.Credentials)
    context = google.datalab.Context(PROJECT, creds)
    self.query = gcm.Query(METRIC_TYPE, context=context)

  @mock.patch('google.datalab.stackdriver.monitoring.Query.iter')
  def test_constructor(self, mock_query_iter):
    time_series_iterable = list(self._query_iter_get_result())
    mock_query_iter.return_value = self._query_iter_get_result()

    query_metadata = gcm.QueryMetadata(self.query)

    mock_query_iter.assert_called_once_with(headers_only=True)
    self.assertEqual(query_metadata.metric_type, METRIC_TYPE)
    self.assertEqual(query_metadata.resource_types, set([RESOURCE_TYPE]))
    self.assertEqual(query_metadata._timeseries_list, time_series_iterable)

  @mock.patch('google.datalab.stackdriver.monitoring.Query.iter')
  def test_iteration(self, mock_query_iter):
    time_series_iterable = list(self._query_iter_get_result())
    mock_query_iter.return_value = self._query_iter_get_result()

    query_metadata = gcm.QueryMetadata(self.query)
    response = list(query_metadata)

    self.assertEqual(len(response), len(time_series_iterable))
    self.assertEqual(response, time_series_iterable)

  @mock.patch('google.datalab.stackdriver.monitoring.Query.iter')
  def test_as_dataframe(self, mock_query_iter):
    mock_query_iter.return_value = self._query_iter_get_result()

    query_metadata = gcm.QueryMetadata(self.query)
    dataframe = query_metadata.as_dataframe()

    NUM_INSTANCES = len(INSTANCE_IDS)

    self.assertEqual(dataframe.shape, (NUM_INSTANCES, 5))

    expected_values = [
        [RESOURCE_TYPE, PROJECT, zone, instance_id, instance_name]
        for zone, instance_id, instance_name
        in zip(INSTANCE_ZONES, INSTANCE_IDS, INSTANCE_NAMES)]
    self.assertEqual(dataframe.values.tolist(), expected_values)

    expected_headers = [
        ('resource.type', ''),
        ('resource.labels', 'project_id'),
        ('resource.labels', 'zone'),
        ('resource.labels', 'instance_id'),
        ('metric.labels', 'instance_name')
    ]
    self.assertEqual(dataframe.columns.tolist(), expected_headers)
    self.assertEqual(dataframe.columns.names, [None, None])

    self.assertEqual(dataframe.index.tolist(), list(range(NUM_INSTANCES)))
    self.assertEqual(dataframe.index.names, [None])

  @mock.patch('google.datalab.stackdriver.monitoring.Query.iter')
  def test_as_dataframe_w_max_rows(self, mock_query_iter):
    mock_query_iter.return_value = self._query_iter_get_result()

    MAX_ROWS = 1
    query_metadata = gcm.QueryMetadata(self.query)
    dataframe = query_metadata.as_dataframe(max_rows=MAX_ROWS)

    self.assertEqual(dataframe.shape, (MAX_ROWS, 5))

    expected_values = [
        [RESOURCE_TYPE, PROJECT, INSTANCE_ZONES[0], INSTANCE_IDS[0],
         INSTANCE_NAMES[0]],
    ]
    self.assertEqual(dataframe.values.tolist(), expected_values)

    expected_headers = [
        ('resource.type', ''),
        ('resource.labels', 'project_id'),
        ('resource.labels', 'zone'),
        ('resource.labels', 'instance_id'),
        ('metric.labels', 'instance_name')
    ]
    self.assertEqual(dataframe.columns.tolist(), expected_headers)
    self.assertEqual(dataframe.columns.names, [None, None])

    self.assertEqual(dataframe.index.tolist(), list(range(MAX_ROWS)))
    self.assertEqual(dataframe.index.names, [None])

  @mock.patch('google.datalab.stackdriver.monitoring.Query.iter')
  def test_as_dataframe_w_no_data(self, mock_query_iter):
    query_metadata = gcm.QueryMetadata(self.query)
    dataframe = query_metadata.as_dataframe()

    self.assertEqual(dataframe.shape, (0, 0))
    self.assertIsNone(dataframe.columns.name)
    self.assertIsNone(dataframe.index.name)

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
