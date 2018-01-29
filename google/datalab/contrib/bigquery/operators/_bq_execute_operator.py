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


class ExecuteOperator(BaseOperator):

  template_fields = ('table', 'parameters', 'path', 'sql')

  @apply_defaults
  def __init__(self, sql, parameters=None, table=None, mode=None, data_source=None, path=None,
               format=None, csv_options=None, schema=None, max_bad_records=None, *args, **kwargs):
    super(ExecuteOperator, self).__init__(*args, **kwargs)
    self.sql = sql
    self.table = table
    self.mode = mode
    self.parameters = parameters
    self.data_source = data_source
    self.path = path
    self.format = format
    self.csv_options = csv_options
    self.schema = schema
    self.max_bad_records = max_bad_records

  def execute(self, context):
    if self.data_source:
      kwargs = {}
      if self.csv_options:
        csv_kwargs = {}
        if 'delimiter' in self.csv_options:
          csv_kwargs['delimiter'] = self.csv_options['delimiter']
        if 'skip' in self.csv_options:
          csv_kwargs['skip_leading_rows'] = self.csv_options['skip']
        if 'strict' in self.csv_options:
          csv_kwargs['allow_jagged_rows'] = self.csv_options['strict']
        if 'quote' in self.csv_options:
          csv_kwargs['quote'] = self.csv_options['quote']
        kwargs['csv_options'] = bq.CSVOptions(**csv_kwargs)

      if self.format:
        kwargs['source_format'] = self.format

      if self.max_bad_records:
        kwargs['max_bad_records'] = self.max_bad_records

      external_data_source = bq.ExternalDataSource(
        source=self.path, schema=bq.Schema(self.schema), **kwargs)
      query = bq.Query(sql=self.sql, data_sources={self.data_source: external_data_source})
    else:
      query = bq.Query(sql=self.sql)

    # use_cache is False since this is most likely the case in pipeline scenarios
    # allow_large_results can be True only if table is specified (i.e. when it's not None)
    kwargs = {}
    if self.mode is not None:
      kwargs['mode'] = self.mode
    output_options = bq.QueryOutput.table(name=self.table, use_cache=False,
                                          allow_large_results=self.table is not None,
                                          **kwargs)
    query_params = bq.Query.get_query_parameters(self.parameters)
    job = query.execute(output_options=output_options, query_params=query_params)

    # Returning the table-name here makes it available for downstream task instances.
    return {
      'table': job.result().full_name
    }
