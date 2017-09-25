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

  @staticmethod
  def _create_context():
    project_id = TestCases.test_project_id
    creds = mock.Mock(spec=google.auth.credentials.Credentials)
    return google.datalab.Context(project_id, creds)

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Table.extract')
  def test_extract_operator(self, mock_table_extract, mock_context_default):
    mock_context_default.return_value = TestCases._create_context()
    cell_args = {'billing': 'foo_billing'}
    extract_operator = ExtractOperator(
      task_id='test_extract_operator', table=TestCases.test_project_id + '.test_table',
      path='test_path', format=None, delimiter=None, header=None, compress=None,
      cell_args=cell_args)

    mock_table_extract.return_value.result = lambda: 'test-results'
    mock_table_extract.return_value.failed = False
    mock_table_extract.return_value.errors = None
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
    expected = """foo = ExecuteOperator(task_id='foo_id', cell_args={'billing': 'test_billing'}, mode='create', query='test_sql', dag=dag)\n"""  # noqa
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
  def test_load_operator(self, mock_table_load, mock_table_exists, mock_api_tables_insert,
                         mock_context_default):
      mock_context_default.return_value = self._create_context()
      cell_args = {'billing': 'test_billing'}

      mock_table_exists.return_value = True
      load_operator = LoadOperator(task_id='test_operator_id', table='project.test.table',
                                   path='test/path', mode='create', format=None, delimiter=None,
                                   skip=None, strict=None, quote=None, schema=None,
                                   cell_args=cell_args)
      with self.assertRaisesRegexp(
              Exception,
              "project.test.table already exists; mode should be \'append\' or \'overwrite\'"):
        load_operator.execute(context=None)

      mock_table_exists.return_value = False
      load_operator = LoadOperator(task_id='test_operator_id', table='project.test.table',
                                   path='test/path', mode='append', format=None, delimiter=None,
                                   skip=None, strict=None, quote=None, schema=None,
                                   cell_args=cell_args)
      with self.assertRaisesRegexp(Exception,
                                   'project.test.table does not exist; mode should be \'create\''):
        load_operator.execute(context=None)

      schema = [
        {"type": "INTEGER", "name": "key"},
        {"type": "FLOAT", "name": "var1"},
        {"type": "FLOAT", "name": "var2"}
      ]
      load_operator = LoadOperator(task_id='test_operator_id', table='project.test.table',
                                   path='test/path', mode='create', format=None, delimiter=None,
                                   skip=None, strict=None, quote=None, schema=schema,
                                   cell_args=cell_args)
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

      mock_table_exists.return_value = True
      load_operator = LoadOperator(task_id='test_operator_id', table='project.test.table',
                                   path='test/path', mode='append', format='csv', delimiter=None,
                                   skip=None, strict=None, quote=None, schema=schema,
                                   cell_args=cell_args)
      load_operator.execute(context=None)
      mock_table_load.assert_called_with('test/path', mode='append',
                                         source_format='csv', csv_options=mock.ANY,
                                         ignore_unknown_values=True)
