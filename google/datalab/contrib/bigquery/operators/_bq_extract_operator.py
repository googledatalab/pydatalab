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


class ExtractOperator(BaseOperator):

  template_fields = ('table', 'path')

  @apply_defaults
  def __init__(self, path, table, format='csv', csv_options=None, *args, **kwargs):
    super(ExtractOperator, self).__init__(*args, **kwargs)
    self.table = table
    self.path = path
    self.format = format
    self.csv_options = csv_options or {}

  def execute(self, context):
    source_table = google.datalab.bigquery.Table(self.table, context=None)

    csv_kwargs = {}
    if 'delimiter' in self.csv_options:
      csv_kwargs['csv_delimiter'] = self.csv_options['delimiter']
    if 'header' in self.csv_options:
      csv_kwargs['csv_header'] = self.csv_options['header']
    if 'compress' in self.csv_options:
      csv_kwargs['compress'] = self.csv_options['compress']

    job = source_table.extract(
      self.path, format='CSV' if self.format == 'csv' else 'NEWLINE_DELIMITED_JSON', **csv_kwargs)

    if job.failed:
      raise Exception('Extract failed: %s' % str(job.fatal_error))
    elif job.errors:
      raise Exception('Extract completed with errors: %s' % str(job.errors))
    return {
      'result': job.result()
    }
