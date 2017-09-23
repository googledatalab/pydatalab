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
import google.datalab.bigquery as bq
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from google.datalab.contrib.pipeline._pipeline import Pipeline


class ExecuteOperator(BaseOperator):

  @apply_defaults
  def __init__(self, sql, udfs=None, data_sources=None, subqueries=None, parameters=None,
               table=None, mode=None, cell_args=None, *args, **kwargs):
    super(ExecuteOperator, self).__init__(*args, **kwargs)
    self._sql = sql
    self._udfs = udfs
    self._data_sources = data_sources
    self._subqueries = subqueries
    self._table = table
    self._mode = mode
    self._parameters = parameters
    self._cell_args = cell_args

  def execute(self, context):
    query = google.datalab.bigquery.Query(sql=self._sql, udfs=self._udfs,
                                          data_sources=self._data_sources,
                                          subqueries=self._subqueries)
    query_params = Pipeline._get_query_parameters(self._parameters)
    output_options = bq.QueryOutput.table(name=self._table, mode=self._mode, use_cache=False,
                                          allow_large_results=True)
    pydatalab_context = google.datalab.bigquery.commands._bigquery._construct_context_for_args(
      self._cell_args)
    query.execute(output_options, context=pydatalab_context, query_params=query_params)