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

import google
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults


class BigQueryExtractOperator(BaseOperator):

  @apply_defaults
  def __init__(self,
               table,
               path,
               format,
               delimiter,
               header,
               compress,
               query,
               view,
               query_params,
               nocache,
               billing,
               *args,
               **kwargs):
    super(BaseOperator, self).__init__(*args, **kwargs)
    self._table = table
    self._path = path
    self._format = format
    self._delimiter = delimiter
    self._header = header
    self._compress = compress
    self._query = query
    self._view = view
    self._query_params = query_params if query else None
    self._nocache = nocache
    self._billing = billing

  def execute(self, context):
      if self._table:
        source_table = google.datalab.bigquery._get_table(self._table)
        if not source_table:
          raise Exception('Could not find table %s' % self._table)

        job = source_table.extract(
            self._path,
            format='CSV' if self._format == 'csv' else 'NEWLINE_DELIMITED_JSON',
            csv_delimiter=self._delimiter, csv_header=self._header,
            compress=self._compress)
      elif self._query or self._view:
        source_name = self._view or self._query
        source = google.datalab.utils.commands.get_notebook_item(source_name)
        if not source:
          raise Exception('Could not find ' +
                          ('view ' + self._view if self._view else 'query ' + self._query))
        query = source if self._query else google.datalab.bigquery.Query.from_view(source)

        output_options = google.datalab.bigquery.QueryOutput.file(
            path=self._path, format=self._format, csv_delimiter=self._delimiter,
            csv_header=self._header, compress=self._compress, use_cache=not self._nocache)
        # TODO(rajivpb): What goes into context?
        job = query.execute(
            output_options,
            context=google.datalab.bigquery._construct_context_for_args({'billing': self._billing}),
            query_params=self._query_params)
      else:
        raise Exception('A query, table, or view is needed to extract')

      if job.failed:
        raise Exception('Extract failed: %s' % str(job.fatal_error))
      elif job.errors:
        raise Exception('Extract completed with errors: %s' % str(job.errors))
      return job.result()