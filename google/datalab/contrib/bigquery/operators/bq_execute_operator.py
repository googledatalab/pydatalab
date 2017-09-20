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
import pickle
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from google.datalab.contrib.pipeline._pipeline import Pipeline


class ExecuteOperator(BaseOperator):

  @apply_defaults
  def __init__(self, query, parameters, table, mode, py_context_str, *args, **kwargs):
    super(ExecuteOperator, self).__init__(*args, **kwargs)
    self._query = query
    self._table = table
    self._mode = mode
    self._parameters = parameters
    self._py_context_str = py_context_str

  def execute(self, context):
    query = google.datalab.utils.commands.get_notebook_item(self._query)
    query_params = Pipeline._get_query_parameters(self._parameters)
    output_options = bq.QueryOutput.table(name=self._table, mode=self._mode, use_cache=False,
                                          allow_large_results=True)
    py_context = pickle.loads(self._py_context_str)
    query.execute(output_options, context=py_context, query_params=query_params)
