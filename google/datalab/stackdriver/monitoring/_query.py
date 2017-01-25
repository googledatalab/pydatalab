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

"""Provides access to metric data as pandas dataframes."""

from __future__ import absolute_import

import google.cloud.monitoring

from . import _query_metadata
from . import _utils


class Query(google.cloud.monitoring.Query):
  """Query object for retrieving metric data."""

  def __init__(self,
               metric_type=google.cloud.monitoring.Query.DEFAULT_METRIC_TYPE,
               end_time=None, days=0, hours=0, minutes=0,
               project_id=None, context=None):
    """Initializes the core query parameters.

    The start time (exclusive) is determined by combining the
    values of ``days``, ``hours``, and ``minutes``, and subtracting
    the resulting duration from the end time.

    It is also allowed to omit the end time and duration here,
    in which case :meth:`~google.cloud.monitoring.query.Query.select_interval`
    must be called before the query is executed.

    Args:
      metric_type: The metric type name. The default value is
          :data:`Query.DEFAULT_METRIC_TYPE
          <google.cloud.monitoring.query.Query.DEFAULT_METRIC_TYPE>`, but
          please note that this default value is provided only for
          demonstration purposes and is subject to change.
      end_time: The end time (inclusive) of the time interval for which
          results should be returned, as a datetime object. The default
          is the start of the current minute.
      days: The number of days in the time interval.
      hours: The number of hours in the time interval.
      minutes: The number of minutes in the time interval.
      project_id: An optional project ID or number to override the one provided
          by the context.
      context: An optional Context object to use instead of the global default.

    Raises:
        ValueError: ``end_time`` was specified but ``days``, ``hours``, and
            ``minutes`` are all zero. If you really want to specify a point in
            time, use
            :meth:`~google.cloud.monitoring.query.Query.select_interval`.
    """
    client = _utils.make_client(project_id, context)
    super(Query, self).__init__(client, metric_type,
                                end_time=end_time,
                                days=days, hours=hours, minutes=minutes)

  def metadata(self):
    """Retrieves the metadata for the query."""
    return _query_metadata.QueryMetadata(self)
