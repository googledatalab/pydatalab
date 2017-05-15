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
import mock
from oauth2client.client import AccessTokenCredentials
import unittest
import json
import pandas
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


import google.datalab  # noqa
import google.datalab.bigquery  # noqa
import google.datalab.bigquery.commands  # noqa
import google.datalab.utils.commands  # noqa


class TestCases(unittest.TestCase):

  @mock.patch('google.datalab.utils.commands.notebook_environment')
  @mock.patch('google.datalab.Context.default')
  def test_udf_cell(self, mock_default_context, mock_notebook_environment):
    env = {}
    mock_default_context.return_value = TestCases._create_context()
    mock_notebook_environment.return_value = env

    # no cell body
    with self.assertRaisesRegexp(Exception, 'UDF return type must be defined'):
      google.datalab.bigquery.commands._bigquery._udf_cell({'name': 'test_udf', 'language': 'js'},
                                                           '')

    # no name
    with self.assertRaisesRegexp(Exception, 'Declaration must be of the form %%bq udf --name'):
      google.datalab.bigquery.commands._bigquery._udf_cell({'name': None, 'language': 'js'},
                                                           'test_cell_body')

    # no return type
    cell_body = """
    // @param word STRING
    // @param corpus STRING
    re = new RegExp(word, 'g');
    return corpus.match(re || []).length;
    """
    with self.assertRaisesRegexp(Exception, 'UDF return type must be defined'):
      google.datalab.bigquery.commands._bigquery._udf_cell({'name': 'count_occurrences',
                                                            'language': 'js'}, cell_body)

    # too many return statements
    cell_body = """
    // @param word STRING
    // @param corpus STRING
    // @returns INTEGER
    // @returns STRING
    re = new RegExp(word, 'g');
    return corpus.match(re || []).length;
    """
    with self.assertRaisesRegexp(Exception, 'Found more than one return'):
      google.datalab.bigquery.commands._bigquery._udf_cell({'name': 'count_occurrences',
                                                            'language': 'js'}, cell_body)

    cell_body = """
    // @param word STRING
    // @param corpus STRING
    // @returns INTEGER
    re = new RegExp(word, 'g');
    return corpus.match(re || []).length;
    """
    google.datalab.bigquery.commands._bigquery._udf_cell({'name': 'count_occurrences',
                                                          'language': 'js'}, cell_body)
    udf = env['count_occurrences']
    self.assertIsNotNone(udf)
    self.assertEquals('count_occurrences', udf._name)
    self.assertEquals('js', udf._language)
    self.assertEquals('INTEGER', udf._return_type)
    self.assertEquals([('word', 'STRING'), ('corpus', 'STRING')], udf._params)
    self.assertEquals([], udf._imports)

  @mock.patch('google.datalab.utils.commands.notebook_environment')
  def test_datasource_cell(self, mock_notebook_env):
    env = {}
    mock_notebook_env.return_value = env
    args = {'name': 'test_ds', 'paths': 'test_path', 'format': None, 'compressed': None}
    cell_body = {
      'schema': [
        {'name': 'col1', 'type': 'int64', 'mode': 'NULLABLE', 'description': 'description1'},
        {'name': 'col1', 'type': 'STRING', 'mode': 'required', 'description': 'description1'}
      ]
    }
    google.datalab.bigquery.commands._bigquery._datasource_cell(args, json.dumps(cell_body))
    self.assertIsInstance(env['test_ds'], google.datalab.bigquery.ExternalDataSource)
    self.assertEqual(env['test_ds']._source, ['test_path'])
    self.assertEqual(env['test_ds']._source_format, 'csv')

  @mock.patch('google.datalab.utils.commands.notebook_environment')
  @mock.patch('google.datalab.Context.default')
  def test_query_cell(self, mock_default_context, mock_notebook_environment):
    env = {}
    mock_default_context.return_value = TestCases._create_context()
    mock_notebook_environment.return_value = env
    IPython.get_ipython().user_ns = env

    # test query creation
    q1_body = 'SELECT * FROM test_table'

    # no query specified
    with self.assertRaises(Exception):
      google.datalab.bigquery.commands._bigquery._query_cell({'name': None, 'udfs': None,
                                                              'datasources': None,
                                                              'subqueries': None}, q1_body)

    # test query creation
    google.datalab.bigquery.commands._bigquery._query_cell({'name': 'q1', 'udfs': None,
                                                            'datasources': None,
                                                            'subqueries': None}, q1_body)
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
  def test_execute_cell(self, mock_get_notebook_item, mock_query_execute, mock_default_context):
    args = {'query': 'test_query', 'verbose': None, 'to_dataframe': None, 'table': None,
            'dataframe_start_row': None, 'dataframe_max_rows': None, 'nocache': None,
            'mode': None, 'large': None}
    cell_body = ''
    mock_get_notebook_item.return_value = google.datalab.bigquery.Query('test_sql')
    google.datalab.bigquery.commands._bigquery._execute_cell(args, cell_body)

    args['to_dataframe'] = True
    google.datalab.bigquery.commands._bigquery._execute_cell(args, cell_body)

    # test --verbose
    args['verbose'] = True
    with mock.patch('sys.stdout', new=StringIO()) as mocked_stdout:
      google.datalab.bigquery.commands._bigquery._execute_cell(args, cell_body)
    self.assertEqual(mocked_stdout.getvalue(), 'test_sql\n')
    args['verbose'] = False

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = AccessTokenCredentials('test_token', 'test_ua')
    return google.datalab.Context(project_id, creds)

  @mock.patch('google.datalab.bigquery.commands._bigquery._get_table')
  @mock.patch('google.datalab.bigquery.Query.execute')
  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.utils.commands.get_notebook_item')
  def test_sample_cell(self, mock_get_notebook_item, mock_context_default,
                       mock_query_execute, mock_get_table):
    args = {'query': None, 'table': None, 'view': None, 'fields': None,
            'count': 5, 'percent': 1, 'key_field': None, 'order': None,
            'profile': None, 'verbose': None, 'method': 'limit'}
    cell_body = ''
    with self.assertRaises(Exception):
      google.datalab.bigquery.commands._bigquery._sample_cell(args, cell_body)

    args['query'] = 'test_query'
    mock_get_notebook_item.return_value = None
    with self.assertRaises(Exception):
      google.datalab.bigquery.commands._bigquery._sample_cell(args, cell_body)

    # query passed, no other parameters
    mock_get_notebook_item.return_value = google.datalab.bigquery.Query('test_sql')
    google.datalab.bigquery.commands._bigquery._sample_cell(args, cell_body)
    call_args = mock_query_execute.call_args[0]
    call_kwargs = mock_query_execute.call_args[1]
    self.assertEqual(call_args[0]._output_type, 'table')
    self.assertEqual(call_kwargs['sampling']('test_sql'),
                     google.datalab.bigquery._sampling.Sampling.default()('test_sql'))

    # test --profile
    args['profile'] = True
    mock_query_execute.return_value.result = lambda: pandas.DataFrame({'c': 'v'}, index=['c'])
    google.datalab.bigquery.commands._bigquery._sample_cell(args, cell_body)
    call_args = mock_query_execute.call_args[0]
    self.assertEqual(call_args[0]._output_type, 'dataframe')

    # test --verbose
    args['verbose'] = True
    with mock.patch('sys.stdout', new=StringIO()) as mocked_stdout:
      google.datalab.bigquery.commands._bigquery._sample_cell(args, cell_body)
    self.assertEqual(mocked_stdout.getvalue(), 'test_sql\n')
    args['verbose'] = False

    # bad query
    mock_get_notebook_item.return_value = None
    with self.assertRaises(Exception):
      google.datalab.bigquery.commands._bigquery._sample_cell(args, cell_body)

    # table passed
    args['query'] = None
    args['table'] = 'test.table'
    mock_get_notebook_item.return_value = google.datalab.bigquery.Table('test.table')
    google.datalab.bigquery.commands._bigquery._sample_cell(args, cell_body)

    # bad table
    mock_get_table.return_value = None
    with self.assertRaises(Exception):
      google.datalab.bigquery.commands._bigquery._sample_cell(args, cell_body)

    # view passed
    args['table'] = None
    args['view'] = 'test_view'
    mock_get_notebook_item.return_value = google.datalab.bigquery.View('test.view')
    google.datalab.bigquery.commands._bigquery._sample_cell(args, cell_body)

    # bad view
    mock_get_notebook_item.return_value = None
    with self.assertRaises(Exception):
      google.datalab.bigquery.commands._bigquery._sample_cell(args, cell_body)

  @mock.patch('google.datalab.bigquery.Query.dry_run')
  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.commands._bigquery._get_query_argument')
  def test_dry_run_cell(self, mock_get_query_argument, mock_context_default, mock_dry_run):
    args = {'query': 'test_query'}
    cell_body = ''
    mock_get_query_argument.return_value = google.datalab.bigquery.Query('test_sql')

    # test --verbose
    args['verbose'] = True
    with mock.patch('sys.stdout', new=StringIO()) as mocked_stdout:
      google.datalab.bigquery.commands._bigquery._dryrun_cell(args, cell_body)
    self.assertEqual(mocked_stdout.getvalue(), 'test_sql\n')
    args['verbose'] = False

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.utils._utils.get_credentials')
  @mock.patch('google.datalab.bigquery.Table.exists')
  @mock.patch('google.datalab.utils.commands.get_notebook_item')
  def test_get_table(self, mock_get_notebook_item, mock_table_exists, mock_get_credentials,
                     mock_default_context):
    # test bad name
    mock_get_notebook_item.return_value = None
    mock_table_exists.return_value = False
    t = google.datalab.bigquery.commands._bigquery._get_table('bad.name')
    self.assertIsNone(t)

    # test good table name
    test_table_name = 'testproject.test.table'
    mock_get_notebook_item.return_value = google.datalab.bigquery.Table(test_table_name)
    t = google.datalab.bigquery.commands._bigquery._get_table(test_table_name)
    self.assertEqual(t.full_name, test_table_name)

    # test table name reference
    mock_get_notebook_item.return_value = test_table_name
    mock_table_exists.return_value = True
    t = google.datalab.bigquery.commands._bigquery._get_table(test_table_name)
    self.assertEqual(t.full_name, test_table_name)

    self.assertIn(test_table_name, google.datalab.bigquery.commands._bigquery._table_cache)

  def test_table_viewer(self):
    # TODO(gram): complete this test
    pass

  @mock.patch('google.datalab.utils._utils.get_credentials')
  @mock.patch('google.datalab.utils._utils.get_default_project_id')
  @mock.patch('google.datalab.utils._utils.save_project_id')
  def test_args_to_context(self, mock_save_project, mock_get_default_project, mock_get_credentials):
    mock_get_credentials.return_value = 'test_creds'
    mock_get_default_project.return_value = 'testproject'

    args = {'billing': 'billing_value'}
    default_context = google.datalab.Context.default()
    c = google.datalab.bigquery.commands._bigquery._construct_context_for_args(args)

    # make sure it's not the same object
    self.assertNotEqual(c, default_context)
    self.assertEqual(c.project_id, default_context.project_id)
    self.assertEqual(c.credentials, default_context.credentials)

    # make sure the right config object was passed
    self.assertEqual(c.config, {'bigquery_billing_tier': 'billing_value'})

    default_context.config['test_prop'] = 'test_val'
    c = google.datalab.bigquery.commands._bigquery._construct_context_for_args(args)
    # make sure other properties in default context were copied
    self.assertEqual(c.config, {'bigquery_billing_tier': 'billing_value', 'test_prop': 'test_val'})

  @mock.patch('google.datalab.utils.commands.get_notebook_item')
  def test_get_query_argument(self, mock_get_notebook_item):
    args = {}
    cell = None
    env = {}
    # an Exception should be raised if no query is specified by name or body
    with self.assertRaises(Exception):
      google.datalab.bigquery.commands._bigquery._get_query_argument(args, cell, env)

    # specify query name, no cell body
    args = {'query': 'test_query'}
    mock_get_notebook_item.return_value = google.datalab.bigquery.Query('test_sql')
    q = google.datalab.bigquery.commands._bigquery._get_query_argument(args, cell, env)
    self.assertEqual(q.sql, 'test_sql')

    # specify query in cell body, no name
    args = {}
    cell = 'test_sql2'
    q = google.datalab.bigquery.commands._bigquery._get_query_argument(args, cell, env)
    self.assertEqual(q.sql, 'test_sql2')

    # specify query by bad name
    args = {'query': 'test_query'}
    mock_get_notebook_item.return_value = None
    with self.assertRaises(Exception):
      google.datalab.bigquery.commands._bigquery._get_query_argument(args, cell, env)

  def test_get_query_parameters(self):
    args = {'query': None}
    cell_body = ''
    with self.assertRaises(Exception):
      params = google.datalab.bigquery.commands._bigquery\
                     ._get_query_parameters(args, json.dumps(cell_body))

    args['query'] = 'test_sql'
    params = google.datalab.bigquery.commands._bigquery\
                   ._get_query_parameters(args, json.dumps(cell_body))
    self.assertEqual(params, {})

    cell_body = {
      'parameters': [
          {'name': 'arg1', 'type': 'INT64', 'value': 5}
      ]
    }

    cell_body['parameters'].append({'name': 'arg2', 'type': 'string', 'value': 'val2'})
    cell_body['parameters'].append({'name': 'arg3', 'type': 'date', 'value': 'val3'})
    params = google.datalab.bigquery.commands._bigquery\
                   ._get_query_parameters(args, json.dumps(cell_body))
    self.assertEqual(len(params), 3)
    self.assertEqual(params, [
      {
        'name': 'arg1',
        'parameterType': {'type': 'INT64'},
        'parameterValue': {'value': 5}
      },
      {
        'name': 'arg2',
        'parameterType': {'type': 'string'},
        'parameterValue': {'value': 'val2'}
      },
      {
        'name': 'arg3',
        'parameterType': {'type': 'date'},
        'parameterValue': {'value': 'val3'}
      }
    ])
