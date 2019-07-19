# Copyright 2016 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.  See the License for the specific language governing permissions and limitations under
# the License.

"""Provides utility methods for the Monitoring API."""

from __future__ import absolute_import

from google.api_core.gapic_v1.client_info import ClientInfo
from google.cloud.monitoring_v3 import MetricServiceClient
from google.cloud.monitoring_v3 import GroupServiceClient

import google.datalab


# _MonitoringClient holds instances of individual google.cloud.monitoring
# clients and translates each call from the old signature, since the prior
# client has been updated and has split into multiple client classes.
class _MonitoringClient(object):
  def __init__(self, context):
    self.project = context.project_id
    client_info = ClientInfo(user_agent='pydatalab/v0')
    self.metrics_client = MetricServiceClient(
      credentials=context.credentials,
      client_info=client_info
    )
    self.group_client = GroupServiceClient(
      credentials=context.credentials,
      client_info=client_info
    )

  def list_metric_descriptors(self, filter_string=None, type_prefix=None):
    filters = []
    if filter_string is not None:
      filters.append(filter_string)

    if type_prefix is not None:
      filters.append('metric.type = starts_with("{prefix}")'.format(
          prefix=type_prefix))

    metric_filter = ' AND '.join(filters)
    metrics = self.metrics_client.list_metric_descriptors(
        self.project, filter_=metric_filter)
    return metrics

  def list_resource_descriptors(self, filter_string=None):
    resources = self.metrics_client.list_monitored_resource_descriptors(
        self.project, filter_=filter_string)
    return resources

  def list_groups(self):
    groups = self.group_client.list_groups(self.project)
    return groups


def make_client(context=None):
  context = context or google.datalab.Context.default()
  client = _MonitoringClient(context)
  return client
