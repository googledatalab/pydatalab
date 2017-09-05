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
from oauth2client.client import AccessTokenCredentials
import json
import mock
import unittest


try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

# import Python so we can mock the parts we need to here.
import IPython
import IPython.core.magic


def noop_decorator(func):
    return func


IPython.core.magic.register_line_cell_magic = noop_decorator
IPython.core.magic.register_line_magic = noop_decorator
IPython.core.magic.register_cell_magic = noop_decorator
IPython.get_ipython = mock.Mock()

import google.datalab
import google.datalab.bigquery
import google.datalab.bigquery.commands
import google.datalab.utils.commands

from google.datalab.contrib.bigquery.operators.bq_extract_operator import ExtractOperator
from google.datalab.contrib.bigquery.operators.bq_execute_operator import ExecuteOperator
from google.datalab.contrib.bigquery.operators.bq_load_operator import LoadOperator


class TestCases(unittest.TestCase):

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = AccessTokenCredentials('test_token', 'test_ua')
    return google.datalab.Context(project_id, creds)

  @mock.patch('google.datalab.bigquery.Table.extract')
  @mock.patch('google.datalab.bigquery.commands._bigquery._get_table')
  def test_extract_operator(self, mock_get_table, mock_table_extract):
    mock_get_table.return_value = None
    extract_operator = ExtractOperator(task_id='test_extract_operator',
      table='test_table', path='test_path', format=None, delimiter=None, header=None,
      compress=None, billing=None)
    with self.assertRaisesRegexp(Exception, 'Could not find table test_table'):
      extract_operator.execute(context=None)

    mock_get_table.return_value = google.datalab.bigquery.Table('project.test.table',
                                                                TestCases._create_context())
    mock_table_extract.return_value.result = lambda: 'test-results'
    mock_table_extract.return_value.failed = False
    mock_table_extract.return_value.errors = None
    self.assertEqual(extract_operator.execute(context=None), 'test-results')
    mock_table_extract.assert_called_with('test_path', format='NEWLINE_DELIMITED_JSON',
                                          csv_delimiter=None, csv_header=None, compress=None)


  @mock.patch('google.datalab.bigquery.Query.execute')
  @mock.patch('google.datalab.utils.commands.notebook_environment')
  @mock.patch('google.datalab.Context.default')
  def test_query_cell(self, mock_default_context, mock_notebook_environment, mock_query_execute):
      env = {}
      mock_default_context.return_value = TestCases._create_context()
      mock_notebook_environment.return_value = env
      IPython.get_ipython().user_ns = env

      # test query creation
      q1_body = 'SELECT * FROM test_table'

      # no query name specified. should execute
      google.datalab.bigquery.commands._bigquery._query_cell({'name': None, 'udfs': None,
                                                              'datasources': None,
                                                              'subqueries': None}, q1_body)
      mock_query_execute.assert_called_with()

      # test query creation
      google.datalab.bigquery.commands._bigquery._query_cell({'name': 'q1', 'udfs': None,
                                                              'datasources': None,
                                                              'subqueries': None}, q1_body)
      mock_query_execute.assert_called_with()

      q1 = env['q1']
      self.assertIsNotNone(q1)
      self.assertEqual(q1.udfs, {})
      self.assertEqual(q1.subqueries, {})
      self.assertEqual(q1_body, q1._sql)
      self.assertEqual(q1_body, q1.sql)

      # test subquery reference and expansion
      q2_body = 'SELECT * FROM q1'
      google.datalab.bigquery.commands._bigquery._query_cell({'name': 'q2', 'udfs': None,
                                                              'datasources': None,
                                                              'subqueries': ['q1']}, q2_body)
      q2 = env['q2']
      self.assertIsNotNone(q2)
      self.assertEqual(q2.udfs, {})
      self.assertEqual({'q1': q1}, q2.subqueries)
      expected_sql = '''\
WITH q1 AS (
  %s
)

%s''' % (q1_body, q2_body)
      self.assertEqual(q2_body, q2._sql)
      self.assertEqual(expected_sql, q2.sql)


  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Query.execute')
  @mock.patch('google.datalab.utils.commands.get_notebook_item')
  def test_execute_operator(self, mock_get_notebook_item, mock_query_execute, mock_default_context):
    mock_default_context.return_value = self._create_context()
    execute_operator = ExecuteOperator(
      task_id='test_execute_operator', query='test_sql', parameters=None, table='test_table',
      mode=None, billing='test_billing')
    output_options = google.datalab.bigquery.QueryOutput.table(
      name='test_table', mode=None, use_cache=False, allow_large_results=True)
    execute_operator.execute(context=None)
    mock_query_execute.return_value.result = lambda: 'test-results'
    mock_query_execute.return_value.failed = False
    mock_query_execute.return_value.errors = None
    mock_query_execute.assert_called_with(
      output_options, context=google.datalab.bigquery.commands._bigquery.
        _construct_context_for_args({'billing': 'test_billing'}), query_params=None)


  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Table.create')
  @mock.patch('google.datalab.bigquery.Table.exists')
  @mock.patch('google.datalab.bigquery.Table.load')
  @mock.patch('google.datalab.bigquery.commands._bigquery._get_table')
  def test_load_cell(self, mock_get_table, mock_table_load, mock_table_exists,
                     mock_table_create, mock_default_context):
      args = {'table': 'project.test.table', 'mode': 'create', 'path': 'test/path', 'skip': None,
              'csv': None, 'delimiter': None, 'format': None, 'strict': None, 'quote': None}
      context = self._create_context()
      table = google.datalab.bigquery.Table('project.test.table')
      mock_get_table.return_value = table
      mock_table_exists.return_value = True
      job = google.datalab.bigquery._query_job.QueryJob('test_id', 'project.test.table',
                                                        'test_sql', context)

      with self.assertRaisesRegexp(Exception, 'already exists; use --append or --overwrite'):
          google.datalab.bigquery.commands._bigquery._load_cell(args, None)

      mock_table_exists.return_value = False

      with self.assertRaisesRegexp(Exception, 'Table does not exist, and no schema specified'):
          google.datalab.bigquery.commands._bigquery._load_cell(args, None)

      cell_body = {
          'schema': [
              {'name': 'col1', 'type': 'int64', 'mode': 'NULLABLE', 'description': 'description1'},
              {'name': 'col1', 'type': 'STRING', 'mode': 'required', 'description': 'description1'}
          ]
      }
      mock_table_load.return_value = job
      job._is_complete = True
      job._fatal_error = 'fatal error'

      with self.assertRaisesRegexp(Exception, 'Load failed: fatal error'):
          google.datalab.bigquery.commands._bigquery._load_cell(args, json.dumps(cell_body))

      job._fatal_error = None
      job._errors = 'error'

      with self.assertRaisesRegexp(Exception, 'Load completed with errors: error'):
          google.datalab.bigquery.commands._bigquery._load_cell(args, json.dumps(cell_body))

      job._errors = None

      google.datalab.bigquery.commands._bigquery._load_cell(args, json.dumps(cell_body))

      mock_table_load.assert_called_with('test/path', mode='create',
                                         source_format='NEWLINE_DELIMITED_JSON',
                                         csv_options=mock.ANY, ignore_unknown_values=True)

      mock_get_table.return_value = None
      mock_table_exists.return_value = True
      args['mode'] = 'append'
      args['format'] = 'csv'

      google.datalab.bigquery.commands._bigquery._load_cell(args, None)

      mock_table_load.assert_called_with('test/path', mode='append',
                                         source_format='csv', csv_options=mock.ANY,
                                         ignore_unknown_values=True)
