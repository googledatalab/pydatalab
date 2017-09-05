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

  @apply_defaults
  def __init__(self, table, path, format, delimiter, header, compress, *args, **kwargs):
    super(ExtractOperator, self).__init__(*args, **kwargs)
    self._table = table
    self._path = path
    self._format = format
    self._delimiter = delimiter
    self._header = header
    self._compress = compress

  def execute(self, context):
      if self._table:
        source_table = google.datalab.bigquery.commands._bigquery._get_table(self._table)
        if not source_table:
          raise Exception('Could not find table %s' % self._table)

        job = source_table.extract(
            self._path,
            format='CSV' if self._format == 'csv' else 'NEWLINE_DELIMITED_JSON',
            csv_delimiter=self._delimiter, csv_header=self._header,
            compress=self._compress)
      else:
        raise Exception('A table is needed to extract')

      if job.failed:
        raise Exception('Extract failed: %s' % str(job.fatal_error))
      elif job.errors:
        raise Exception('Extract completed with errors: %s' % str(job.errors))
      return job.result()
