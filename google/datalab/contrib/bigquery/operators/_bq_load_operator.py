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


class LoadOperator(BaseOperator):
  """Implements the BigQuery load magic used to load data from GCS to a table.
   The supported syntax is:
       %bq load <optional args>
  Args:
    args: the arguments following '%bq load'.
    cell_body: optional contents of the cell interpreted as YAML or JSON.
  Returns:
    A message about whether the load succeeded or failed.
  """

  template_fields = ('_table', '_path')

  @apply_defaults
  def __init__(self, table, path, mode='append', format='csv', schema=None, csv_options=None, *args,
               **kwargs):
    super(LoadOperator, self).__init__(*args, **kwargs)
    self._table = table
    self._path = path
    self._mode = mode
    self._format = format
    self._csv_options = csv_options or {}
    self._schema = schema

  def execute(self, context):
    if self._table:
      table = bq.Table(self._table, context=None)

    if self._mode == 'create':
      if table.exists():
        raise Exception(
          "%s already exists; mode should be \'append\' or \'overwrite\'" % self._table)
      if not self._schema:
        raise Exception(
          '%s does not exist, and no schema specified in cell; cannot load.' % self._table)
      table.create(schema=self._schema)
    elif not table.exists():
      raise Exception('%s does not exist; mode should be \'create\'' % self._table)

    csv_options = bq.CSVOptions(
      delimiter=self._csv_options.get('delimiter'), skip_leading_rows=self._csv_options.get('skip'),
      allow_jagged_rows=self._csv_options.get('strict'), quote=self._csv_options.get('quote'))
    job = table.load(self._path, mode=self._mode,
                     source_format=('csv' if self._format == 'csv' else 'NEWLINE_DELIMITED_JSON'),
                     csv_options=csv_options,
                     ignore_unknown_values=not self._csv_options.get('strict'))
    if job.failed:
      raise Exception('Load failed: %s' % str(job.fatal_error))
    elif job.errors:
      raise Exception('Load completed with errors: %s' % str(job.errors))
