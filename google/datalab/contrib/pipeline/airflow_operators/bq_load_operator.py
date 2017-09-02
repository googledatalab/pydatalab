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


class BigQueryLoadOperator(BaseOperator):
  """Implements the BigQuery load magic used to load data from GCS to a table.
   The supported syntax is:
       %bq load <optional args>
  Args:
    args: the arguments following '%bq load'.
    cell_body: optional contents of the cell interpreted as YAML or JSON.
  Returns:
    A message about whether the load succeeded or failed.
  """
  @apply_defaults
  def __init__(self,
               table,
               path,
               mode,
               format,
               delimiter,
               skip,
               strict,
               quote,
               schema=None,
               *args,
               **kwargs):
    super(BaseOperator, self).__init__(*args, **kwargs)
    self.table = table
    self.path = path
    self.schema = schema
    self.mode = mode
    self.delimiter = delimiter
    self.format = format
    self.skip = skip
    self.strict = strict
    self.quote = quote

  def execute(self, context):
    bq_table = google.datalab.bigquery._get_table(self.table)
    if not bq_table:
      bq_table = google.datalab.bigquery.Table(self.table)

    if self.schema:
      schema = google.datalab.bigquery.Schema(self.schema)
      bq_table.create(schema=schema)
    elif bq_table.exists():
      if self.mode == 'create':
        raise Exception('%s already exists; use --append or --overwrite' % self.table)
    else:
      raise Exception('Table does not exist, and no schema specified in cell; cannot load')

    csv_options = google.datalab.bigquery.CSVOptions(delimiter=self.delimiter,
                                                     skip_leading_rows=self.skip,
                                                     allow_jagged_rows=self.strict,
                                                     quote=self.quote)
    job = bq_table.load(self.path, mode=self.mode,
                        source_format=('csv' if self.format == 'csv' else 'NEWLINE_DELIMITED_JSON'),
                        csv_options=csv_options, ignore_unknown_values=not self.strict)
    if job.failed:
      raise Exception('Load failed: %s' % str(job.fatal_error))
    elif job.errors:
      raise Exception('Load completed with errors: %s' % str(job.errors))
