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

"""QueryMetadata object that shows the metadata in a query's results."""


from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import object

from google.cloud.monitoring_v3 import _dataframe
from google.protobuf.json_format import MessageToDict
import pandas


class QueryMetadata(object):
  """QueryMetadata object contains the metadata of a timeseries query."""

  def __init__(self, query):
    """Initializes the QueryMetadata given the query object.

    Args:
      query: A Query object.
    """
    self._timeseries_list = list(query.iter(headers_only=True))

    # Note: If self._timeseries_list has even one entry, the metric type
    # can be extracted from there as well.
    self._metric_type = query.metric_type

  def __iter__(self):
    for timeseries in self._timeseries_list:
      yield timeseries

  @property
  def metric_type(self):
    """Returns the metric type in the underlying query."""
    return self._metric_type

  @property
  def resource_types(self):
    """Returns a set containing resource types in the query result."""
    return set([ts.resource.type for ts in self._timeseries_list])

  def as_dataframe(self, max_rows=None):
      """Creates a pandas dataframe from the query metadata.

      Args:
        max_rows: The maximum number of timeseries metadata to return. If None,
            return all.

      Returns:
        A pandas dataframe containing the resource type, resource labels and
        metric labels. Each row in this dataframe corresponds to the metadata
        from one time series.
      """
      max_rows = len(self._timeseries_list) if max_rows is None else max_rows
      headers = [{
          'resource': MessageToDict(ts.resource),
          'metric': MessageToDict(ts.metric)
      } for ts in self._timeseries_list[:max_rows]]

      if not headers:
        return pandas.DataFrame()

      dataframe = pandas.io.json.json_normalize(headers)

      # Add a 2 level column header.
      dataframe.columns = pandas.MultiIndex.from_tuples(
          [(col, '') if col == 'resource.type' else col.rsplit('.', 1)
           for col in dataframe.columns])

      # Re-order the columns.
      resource_keys = _dataframe._sorted_resource_labels(
          dataframe['resource.labels'].columns)
      sorted_columns = [('resource.type', '')]
      sorted_columns += [('resource.labels', key) for key in resource_keys]
      sorted_columns += sorted(col for col in dataframe.columns
                               if col[0] == 'metric.labels')
      dataframe = dataframe[sorted_columns]

      # Sort the data, and clean up index values, and NaNs.
      dataframe = dataframe.sort_values(sorted_columns)
      dataframe = dataframe.reset_index(drop=True).fillna('')
      return dataframe
