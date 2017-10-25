# Copyright 2017 Google Inc. All rights reserved.
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

import google.datalab.bigquery as bq
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from google.datalab.contrib.pipeline._pipeline import Pipeline


class ExecuteOperator(BaseOperator):

  template_fields = ('_sql', '_table')

  @apply_defaults
  def __init__(self, sql, parameters=None, table=None, mode=None, data_source=None, path=None,
               format=None, csv_options=None, schema=None, max_bad_records=None, *args, **kwargs):
    super(ExecuteOperator, self).__init__(*args, **kwargs)
    self._sql = sql
    self._table = table
    self._mode = mode
    self._parameters = parameters
    self._data_source = data_source
    self._path = path
    self._format = format
    self._csv_options = csv_options or {}
    self._schema = schema
    self._max_bad_records = max_bad_records

  def execute(self, context):
    kwargs = {}
    if self._data_source:
      if self._csv_options and self._csv_options.__len__() > 1:
        csv_kwargs = {}
        if 'delimiter' in self._csv_options:
          csv_kwargs['delimiter'] = self._csv_options['delimiter']
        if 'skip' in self._csv_options:
          csv_kwargs['skip_leading_rows'] = self._csv_options['skip']
        if 'strict' in self._csv_options:
          csv_kwargs['allow_jagged_rows'] = self._csv_options['strict']
        if 'quote' in self._csv_options:
          csv_kwargs['quote'] = self._csv_options['quote']
          kwargs['csv_options'] = bq.CSVOptions(**csv_kwargs)

      if self._format:
        kwargs['source_format'] = self._format

      if self._max_bad_records:
        kwargs['max_bad_records'] = self._max_bad_records

      external_data_source = bq.ExternalDataSource(
        source=self._path, schema=bq.Schema(self._schema), **kwargs)
      query = bq.Query(sql=self._sql, data_sources={self._data_source: external_data_source})
    else:
      query = bq.Query(sql=self._sql)

    # use_cache is False since this is most likely the case in pipeline scenarios
    # allow_large_results can be True only if table is specified (i.e. when it's not None)
    output_options = bq.QueryOutput.table(name=self._table, mode=self._mode, use_cache=False,
                                          allow_large_results=self._table is not None)

    query_params = Pipeline._get_query_parameters(self._parameters)
    job = query.execute(output_options, query_params=query_params)

    # Returning the table-name here makes it available for downstream task instances.
    return {
      'table': job.result().name
    }
