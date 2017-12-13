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

import google.datalab.contrib.pipeline._pipeline as pipeline
import mock
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

  test_project_id = 'test_project'
  test_table_name = 'project.test.table'
  test_schema = [
      {"type": "INTEGER", "name": "key"},
      {"type": "FLOAT", "name": "var1"},
      {"type": "FLOAT", "name": "var2"}
  ]

  @staticmethod
  def _create_context():
    project_id = TestCases.test_project_id
    creds = mock.Mock(spec=google.auth.credentials.Credentials)
    return google.datalab.Context(project_id, creds)

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Table.extract')
  def test_extract_operator(self, mock_table_extract, mock_context_default):
    mock_context_default.return_value = TestCases._create_context()
    extract_operator = ExtractOperator(table=TestCases.test_project_id + '.test_table',
                                       path='test_path', format=None,
                                       task_id='test_extract_operator')

    # Happy path
    mock_table_extract.return_value.result = lambda: 'test-results'
    mock_table_extract.return_value.failed = False
    mock_table_extract.return_value.errors = None
    self.assertDictEqual(extract_operator.execute(context=None), {'result': 'test-results'})
    mock_table_extract.assert_called_with('test_path', format='NEWLINE_DELIMITED_JSON',
                                          csv_delimiter=None, csv_header=None, compress=None)

    # Extract failed
    mock_table_extract.return_value.result = lambda: 'test-results'
    mock_table_extract.return_value.failed = True
    mock_table_extract.return_value.errors = None
    with self.assertRaisesRegexp(Exception, "Extract failed:"):
      extract_operator.execute(context=None)

    # Extract completed with errors
    mock_table_extract.return_value.result = lambda: 'test-results'
    mock_table_extract.return_value.failed = False
    mock_table_extract.return_value.errors = 'foo_error'
    with self.assertRaisesRegexp(Exception, 'Extract completed with errors: foo_error'):
      extract_operator.execute(context=None)

  @mock.patch('google.datalab.bigquery.Query.execute')
  @mock.patch('google.datalab.utils.commands.get_notebook_item')
  def test_execute_operator_definition(self, mock_get_notebook_item, mock_query_execute):
    mock_get_notebook_item.return_value = google.datalab.bigquery.Query('test_sql')
    task_id = 'foo'
    task_details = {}
    task_details['type'] = 'pydatalab.bq.execute'
    task_details['sql'] = 'test_sql'
    task_details['mode'] = 'create'

    actual = pipeline.Pipeline(None, None)._get_operator_definition(task_id, task_details, None)
    expected = """foo = ExecuteOperator(task_id='foo_id', mode=\"\"\"create\"\"\", sql=\"\"\"test_sql\"\"\", dag=dag)
"""  # noqa
    self.assertEqual(actual, expected)

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Query.execute')
  @mock.patch('google.datalab.bigquery.QueryOutput.table')
  @mock.patch('google.datalab.bigquery._query_job.QueryJob')
  def test_execute_operator(self, mock_query_job, mock_query_output_table, mock_query_execute,
                            mock_context_default):
    mock_context_default.return_value = self._create_context()

    execute_operator = ExecuteOperator(task_id='test_execute_operator', sql='test_sql')
    mock_query_execute.return_value = mock_query_job
    mock_query_job.result.return_value = google.datalab.bigquery.Table(TestCases.test_table_name)
    self.assertDictEqual(execute_operator.execute(context=None),
                         {'table': TestCases.test_table_name})
    mock_query_output_table.assert_called_with(name=None, mode=None, use_cache=False,
                                               allow_large_results=False)

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.ExternalDataSource')
  @mock.patch('google.datalab.bigquery.Query')
  @mock.patch('google.datalab.bigquery.QueryOutput.table')
  @mock.patch('google.datalab.bigquery._query_job.QueryJob')
  def test_execute_operator_with_data_source(self, mock_query_job, mock_query_output_table,
                                             mock_query_class, mock_external_data_source,
                                             mock_context_default):
    mock_context_default.return_value = self._create_context()
    csv_options = {'delimiter': 'f', 'skip': 9, 'strict': True, 'quote': 'l'}
    execute_operator = ExecuteOperator(task_id='test_execute_operator', sql='test_sql',
                                       data_source='foo_data_source', path='foo_path',
                                       max_bad_records=20, schema=TestCases.test_schema,
                                       csv_options=csv_options, format='csv')
    mock_query_instance = mock_query_class.return_value
    mock_query_instance.execute.return_value = mock_query_job
    mock_query_job.result.return_value = google.datalab.bigquery.Table(TestCases.test_table_name)
    self.assertDictEqual(execute_operator.execute(context=None),
                         {'table': TestCases.test_table_name})
    mock_query_output_table.assert_called_with(name=None, mode=None, use_cache=False,
                                               allow_large_results=False)
    mock_query_class.assert_called_with(
        sql='test_sql', data_sources={'foo_data_source': mock_external_data_source.return_value})
    mock_external_data_source.assert_called_with(
      source='foo_path', max_bad_records=20, csv_options=mock.ANY, source_format='csv',
      schema=google.datalab.bigquery.Schema(TestCases.test_schema))

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery._api.Api.tables_insert')
  @mock.patch('google.datalab.bigquery.Table.create')
  @mock.patch('google.datalab.bigquery.Table.exists')
  @mock.patch('google.datalab.bigquery.Table.load')
  def test_load_operator(self, mock_table_load, mock_table_exists, mock_table_create,
                         mock_api_tables_insert, mock_context_default):
      mock_context_default.return_value = self._create_context()

      # Table exists
      mock_table_exists.return_value = True
      load_operator = LoadOperator(table=TestCases.test_table_name, path='test/path', mode='append',
                                   format=None, csv_options=None, schema=None,
                                   task_id='test_operator_id')
      mock_job = mock.Mock()
      mock_job.result.return_value = 'test-result'
      mock_job.failed = False
      mock_job.errors = False
      mock_table_load.return_value = mock_job
      load_operator.execute(context=None)
      mock_table_load.assert_called_with('test/path', mode='append',
                                         source_format='NEWLINE_DELIMITED_JSON',
                                         csv_options=mock.ANY, ignore_unknown_values=True)

      # Table does not exist
      mock_table_exists.return_value = False
      csv_options = {'delimiter': 'f', 'skip': 9, 'strict': True, 'quote': '"'}
      schema = [
        {"type": "INTEGER", "name": "key"},
        {"type": "FLOAT", "name": "var1"},
        {"type": "FLOAT", "name": "var2"}
      ]
      load_operator = LoadOperator(table=TestCases.test_table_name, path='test/path', mode='append',
                                   format=None, csv_options=csv_options, schema=schema,
                                   task_id='test_operator_id')
      mock_job = mock.Mock()
      mock_job.result.return_value = 'test-result'
      mock_job.failed = False
      mock_job.errors = False
      mock_table_load.return_value = mock_job
      load_operator.execute(context=None)
      mock_table_load.assert_called_with('test/path', mode='append',
                                         source_format='NEWLINE_DELIMITED_JSON',
                                         csv_options=mock.ANY, ignore_unknown_values=False)
      mock_table_create.assert_called_with(schema=schema)

      # Table load fails
      load_operator = LoadOperator(table=TestCases.test_table_name, path='test/path', mode='append',
                                   format=None, csv_options=None, schema=schema,
                                   task_id='test_operator_id')
      mock_job = mock.Mock()
      mock_job.failed = True
      mock_job.fatal_error = 'fatal error'
      mock_table_load.return_value = mock_job
      with self.assertRaisesRegexp(Exception, 'Load failed: fatal error'):
        load_operator.execute(context=None)

      # Table load completes with errors
      load_operator = LoadOperator(table=TestCases.test_table_name, path='test/path', mode='append',
                                   format=None, csv_options=None, schema=TestCases.test_schema,
                                   task_id='test_operator_id')
      mock_job = mock.Mock()
      mock_job.failed = False
      mock_job.errors = 'error'
      mock_table_load.return_value = mock_job
      with self.assertRaisesRegexp(Exception, 'Load completed with errors: error'):
        load_operator.execute(context=None)

  def test_execute_operator_defaults(self):
    execute_operator = ExecuteOperator(task_id='foo_task_id', sql='foo_sql')
    self.assertIsNone(execute_operator.parameters)
    self.assertIsNone(execute_operator.table)
    self.assertIsNone(execute_operator.mode)
    self.assertEqual(execute_operator.template_fields, ('table', 'parameters', 'path'))

  def test_extract_operator_defaults(self):
    extract_operator = ExtractOperator(task_id='foo_task_id', path='foo_path', table='foo_table')
    self.assertEquals(extract_operator.format, 'csv')
    self.assertDictEqual(extract_operator.csv_options, {})
    self.assertEqual(extract_operator.template_fields, ('table', 'path'))

  def test_load_operator_defaults(self):
    load_operator = LoadOperator(task_id='foo_task_id', path='foo_path', table='foo_table')
    self.assertEquals(load_operator.format, 'csv')
    self.assertEquals(load_operator.mode, 'append')
    self.assertIsNone(load_operator.schema)
    self.assertDictEqual(load_operator.csv_options, {})
    self.assertEqual(load_operator.template_fields, ('table', 'path'))
