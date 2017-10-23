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

  template_fields = ('_table', '_path')

  @apply_defaults
  def __init__(self, path, table=None, format='csv', csv_options=None, *args, **kwargs):
    super(ExtractOperator, self).__init__(*args, **kwargs)
    self._table = table
    self._path = path
    self._format = format
    self._csv_options = csv_options or {}

  def execute(self, context):
    if not self._table:
      task_instance = context['task_instance']
      # If the table is not specified, we fetch it from the output of the execute task, i.e. the
      # query results table. This could either be a permanent table or a temporary table. If we're
      # here, it is most likely a temporary table. If it was a permanent one, it would have been
      # passed in as a param and we wouldn't be here.
      # TODO(rajivpb): Assert that if we're here, then the table is a temporary one.
      # TODO(rajivpb):
      # The task id of the execute task is created by concatenating 'bq_pipeline_execute_task'
      # and '_id'. This is currently being hard-coded, but consider making this a parameter.
      # It could require substantial changes to the underlying object model of Pipeline.
      execute_task_output = task_instance.xcom_pull(task_ids='bq_pipeline_execute_task_id')
      self._table = execute_task_output.get('table')

    source_table = google.datalab.bigquery.Table(self._table, context=None)
    job = source_table.extract(
      self._path, format='CSV' if self._format == 'csv' else 'NEWLINE_DELIMITED_JSON',
      csv_delimiter=self._csv_options.get('delimiter'),
      csv_header=self._csv_options.get('header'), compress=self._csv_options.get('compress'))

    if job.failed:
      raise Exception('Extract failed: %s' % str(job.fatal_error))
    elif job.errors:
      raise Exception('Extract completed with errors: %s' % str(job.errors))
    return {
      'result': job.result()
    }
