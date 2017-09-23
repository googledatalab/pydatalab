# Copyright 2015 Google Inc. All rights reserved.
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

from __future__ import absolute_import
from __future__ import unicode_literals
import google.datalab.contrib.pipeline._pipeline as pipeline
import mock
import pickle
import re
import unittest


# import Python so we can mock the parts we need to here.
import IPython.core.magic


def noop_decorator(func):
    return func


IPython.core.magic.register_line_cell_magic = noop_decorator
IPython.core.magic.register_line_magic = noop_decorator
IPython.core.magic.register_cell_magic = noop_decorator
IPython.get_ipython = mock.Mock()

import google.datalab  # noqa
import google.datalab.bigquery  # noqa
import google.datalab.bigquery.commands  # noqa
import google.datalab.utils.commands  # noqa

from google.datalab.contrib.bigquery.operators._bq_extract_operator import ExtractOperator  # noqa
from google.datalab.contrib.bigquery.operators._bq_execute_operator import ExecuteOperator  # noqa
from google.datalab.contrib.bigquery.operators._bq_load_operator import LoadOperator  # noqa


class TestCases(unittest.TestCase):

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = mock.Mock(spec=google.auth.credentials.Credentials)
    return google.datalab.Context(project_id, creds)

  @mock.patch('google.datalab.bigquery.Table.extract')
  def test_extract_operator(self, mock_table_extract):
    cell_args = {'billing': 'foo_billing'}
    extract_operator = ExtractOperator(
      task_id='test_extract_operator',table='test_project.test_table', path='test_path',
      format=None, delimiter=None, header=None, compress=None, cell_args=cell_args)

    mock_table_extract.return_value.result = lambda: 'test-results'
    #mock_table_extract.return_value.failed = False
    #mock_table_extract.return_value.errors = None
    self.assertEqual(extract_operator.execute(context=None), 'test-results')
    mock_table_extract.assert_called_with('test_path', format='NEWLINE_DELIMITED_JSON',
                                          csv_delimiter=None, csv_header=None, compress=None)

  @mock.patch('google.datalab.bigquery.Query.execute')
  @mock.patch('google.datalab.utils.commands.get_notebook_item')
  def test_execute_operator_definition(self, mock_get_notebook_item, mock_query_execute):
    mock_get_notebook_item.return_value = google.datalab.bigquery.Query('test_sql')
    task_id = 'foo'
    task_details = {}
    task_details['type'] = 'pydatalab.bq.execute'
    task_details['query'] = 'test_sql'
    task_details['mode'] = 'create'
    task_details['cell_args'] = {'billing': 'test_billing'}

    actual = pipeline.Pipeline(None, None)._get_operator_definition(task_id, task_details)
    expected = """foo = ExecuteOperator(task_id='foo_id', cell_args={u'billing': u'test_billing'}, mode='create', query='test_sql', dag=dag)\n"""  # noqa
    self.assertEqual(actual, expected)

  @mock.patch('google.datalab.bigquery.Query.execute')
  @mock.patch('google.datalab.utils.commands.get_notebook_item')
  def execute_operator(self, mock_get_notebook_item, mock_query_execute):
    mock_get_notebook_item.return_value = google.datalab.bigquery.Query('test_sql')
    cell_args = {'billing': 'test_billing'}
    # This statement is required even though it seems like it's not. Go figure.
    execute_operator = ExecuteOperator(
      task_id='test_execute_operator', query='test_sql', parameters=None, table='test_table',
      mode=None, **cell_args)
    execute_operator.execute(context=None)
    # TODO(rajivpb): Mock output_options and query_params for a more complete test.
    mock_query_execute.assert_called_once()

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery._api.Api.tables_insert')
  @mock.patch('google.datalab.bigquery.Table.exists')
  @mock.patch('google.datalab.bigquery.Table.load')
  @mock.patch('google.datalab.bigquery.commands._bigquery._get_table')
  def test_load_operator(self, mock_get_table, mock_table_load, mock_table_exists,
                         mock_api_tables_insert, mock_context_default):
      load_operator = LoadOperator(task_id='test_operator_id', table='project.test.table',
                                   path='test/path', mode='create', format=None, delimiter=None,
                                   skip=None, strict=None, quote=None, schema=None)
      mock_context_default.return_value = self._create_context()
      table = google.datalab.bigquery.Table('project.test.table', mock_context_default)
      mock_get_table.return_value = table
      mock_table_exists.return_value = True

      with self.assertRaisesRegexp(Exception, 'already exists; use --append or --overwrite'):
        load_operator.execute(context=None)

      mock_table_exists.return_value = False

      with self.assertRaisesRegexp(Exception, 'Table does not exist, and no schema specified'):
        load_operator.execute(context=None)

      schema = [
          {'name': 'col1', 'type': 'int64', 'mode': 'NULLABLE', 'description': 'description1'},
          {'name': 'col1', 'type': 'STRING', 'mode': 'required', 'description': 'description1'}
      ]
      load_operator = LoadOperator(task_id='test_operator_id', table='project.test.table',
                                   path='test/path', mode='create', format=None, delimiter=None,
                                   skip=None, strict=None, quote=None, schema=schema)
      job = google.datalab.bigquery._query_job.QueryJob('test_id', 'project.test.table',
                                                        'test_sql', None)
      mock_table_load.return_value = job
      job._is_complete = True
      job._fatal_error = 'fatal error'
      mock_api_tables_insert.return_value = {'selfLink': 'http://foo'}
      with self.assertRaisesRegexp(Exception, 'Load failed: fatal error'):
        load_operator.execute(context=None)

      job._fatal_error = None
      job._errors = 'error'
      with self.assertRaisesRegexp(Exception, 'Load completed with errors: error'):
        load_operator.execute(context=None)

      job._errors = None
      load_operator.execute(context=None)
      mock_table_load.assert_called_with('test/path', mode='create',
                                         source_format='NEWLINE_DELIMITED_JSON',
                                         csv_options=mock.ANY, ignore_unknown_values=True)

      mock_get_table.return_value = None
      mock_table_exists.return_value = True
      load_operator = LoadOperator(task_id='test_operator_id', table='project.test.table',
                                   path='test/path', mode='append', format='csv', delimiter=None,
                                   skip=None, strict=None, quote=None, schema=schema)
      load_operator.execute(context=None)
      mock_table_load.assert_called_with('test/path', mode='append',
                                         source_format='csv', csv_options=mock.ANY,
                                         ignore_unknown_values=True)
