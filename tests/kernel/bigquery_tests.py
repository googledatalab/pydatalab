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
import json
import mock
import pandas
import six
import unittest

from datetime import datetime

import google.auth
import google.datalab  # noqa
import google.datalab.bigquery as bq
import google.datalab.bigquery.commands
import google.datalab.utils.commands  # noqa
# import Python so we can mock the parts we need to here.
import IPython
import IPython.core.magic


def noop_decorator(func):
  return func


IPython.core.magic.register_line_cell_magic = noop_decorator
IPython.core.magic.register_line_magic = noop_decorator
IPython.core.magic.register_cell_magic = noop_decorator
IPython.get_ipython = mock.Mock()


class TestCases(unittest.TestCase):

  def test_table_schema(self):
    import jsonschema
    good_schemas = [
      {
        'schema': [
          {'name': 'col1', 'type': 'int64', 'mode': 'NULLABLE', 'description': 'description1'},
          {'name': 'col1', 'type': 'STRING', 'mode': 'required', 'description': 'description1'}
        ]
      },
      {
        'schema': [
          {'name': 'col1', 'type': 'record', 'mode': 'repeated', 'description': 'description1',
           'fields': [
             {'name': 'field1', 'type': 'int64'},
             {'name': 'field2', 'type': 'int64'},
             {'name': 'field3', 'type': 'record',
              'fields': [
                {'name': 'nestedField1', 'type': 'STRING'},
                {'name': 'nestedField2', 'type': 'STRING'}
              ]}
           ]}
        ]
      }
    ]

    bad_schemas = [
      {
        # Bad type.
        'schema': [
          {'name': 'col1', 'type': 'badtype'}
        ]
      },
      {
        # Bad type. Strictly upper and lower case are supported.
        'schema': [
          {'name': 'col1', 'type': 'stRING'}
        ]
      },
      {
        # Missing name.
        'schema': [
          {'type': 'string'}
        ]
      },
      {
        # Missing type.
        'schema': [
          {'name': 'col1'}
        ]
      },
      {
        # Fields should be an array.
        'schema': [
          {'name': 'col1', 'type': 'string', 'fields': 'badfields'}
        ]
      }
    ]

    for s in good_schemas:
      record = google.datalab.utils.commands.parse_config(json.dumps(s), {})
      jsonschema.validate(record,
                          bq.commands._bigquery.BigQuerySchema.TABLE_SCHEMA_SCHEMA)

    for s in bad_schemas:
      record = google.datalab.utils.commands.parse_config(json.dumps(s), {})
      with self.assertRaises(Exception):
        jsonschema.validate(record, bq.commands._bigquery.table_schema_schema)

  @mock.patch('google.datalab.utils.commands.notebook_environment')
  @mock.patch('google.datalab.Context.default')
  def test_udf_cell(self, mock_default_context, mock_notebook_environment):
    env = {}
    mock_default_context.return_value = TestCases._create_context()
    mock_notebook_environment.return_value = env

    # no cell body
    with self.assertRaisesRegexp(Exception, 'UDF return type must be defined'):
      bq.commands._bigquery._udf_cell({'name': 'test_udf', 'language': 'js'}, '')

    # no name
    with self.assertRaisesRegexp(Exception, 'Declaration must be of the form %%bq udf --name'):
      bq.commands._bigquery._udf_cell({'name': None, 'language': 'js'}, 'test_cell_body')

    # no return type
    cell_body = """
    // @param word STRING
    // @param corpus STRING
    re = new RegExp(word, 'g');
    return corpus.match(re || []).length;
    """
    with self.assertRaisesRegexp(Exception, 'UDF return type must be defined'):
      bq.commands._bigquery._udf_cell({'name': 'count_occurrences', 'language': 'js'}, cell_body)

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
      bq.commands._bigquery._udf_cell({'name': 'count_occurrences', 'language': 'js'}, cell_body)

    cell_body = """
    // @param word STRING
    // @param corpus STRING
    // @returns INTEGER
    re = new RegExp(word, 'g');
    return corpus.match(re || []).length;
    """
    bq.commands._bigquery._udf_cell({'name': 'count_occurrences', 'language': 'js'}, cell_body)
    udf = env['count_occurrences']
    self.assertIsNotNone(udf)
    self.assertEquals('count_occurrences', udf._name)
    self.assertEquals('js', udf._language)
    self.assertEquals('INTEGER', udf._return_type)
    self.assertEquals([('word', 'STRING'), ('corpus', 'STRING')], udf._params)
    self.assertEquals([], udf._imports)

    # param types with spaces (regression for pull request 373)
    cell_body = """
    // @param test_param ARRAY<STRUCT<index INT64, value STRING>>
    // @returns INTEGER
    """
    bq.commands._bigquery._udf_cell({'name': 'count_occurrences', 'language': 'js'}, cell_body)
    udf = env['count_occurrences']
    self.assertIsNotNone(udf)
    self.assertEquals([('test_param', 'ARRAY<STRUCT<index INT64, value STRING>>')], udf._params)

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
    bq.commands._bigquery._datasource_cell(args, json.dumps(cell_body))
    self.assertIsInstance(env['test_ds'], bq.ExternalDataSource)
    self.assertEqual(env['test_ds']._source, ['test_path'])
    self.assertEqual(env['test_ds']._source_format, 'csv')

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
    bq.commands._bigquery._query_cell(
      {'name': None, 'udfs': None, 'datasources': None, 'subqueries': None}, q1_body)
    mock_query_execute.assert_called_with()

    # test query creation
    bq.commands._bigquery._query_cell(
      {'name': 'q1', 'udfs': None, 'datasources': None, 'subqueries': None}, q1_body)
    mock_query_execute.assert_called_with()

    q1 = env['q1']
    self.assertIsNotNone(q1)
    self.assertEqual(q1.udfs, {})
    self.assertEqual(q1.subqueries, {})
    self.assertEqual(q1_body, q1._sql)
    self.assertEqual(q1_body, q1.sql)

    # test subquery reference and expansion
    q2_body = 'SELECT * FROM q1'
    bq.commands._bigquery._query_cell(
      {'name': 'q2', 'udfs': None, 'datasources': None, 'subqueries': ['q1']}, q2_body)
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
    mock_get_notebook_item.return_value = bq.Query('test_sql')
    bq.commands._bigquery._execute_cell(args, cell_body)

    args['to_dataframe'] = True
    bq.commands._bigquery._execute_cell(args, cell_body)

    # test --verbose
    args['verbose'] = True
    with mock.patch('sys.stdout', new=six.StringIO()) as mocked_stdout:
      bq.commands._bigquery._execute_cell(args, cell_body)
    self.assertEqual(mocked_stdout.getvalue(), 'test_sql\n')
    args['verbose'] = False

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = mock.Mock(spec=google.auth.credentials.Credentials)
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
      bq.commands._bigquery._sample_cell(args, cell_body)

    args['query'] = 'test_query'
    mock_get_notebook_item.return_value = None
    with self.assertRaises(Exception):
      bq.commands._bigquery._sample_cell(args, cell_body)

    # query passed, no other parameters
    mock_get_notebook_item.return_value = bq.Query('test_sql')
    bq.commands._bigquery._sample_cell(args, cell_body)
    call_args = mock_query_execute.call_args[0]
    call_kwargs = mock_query_execute.call_args[1]
    self.assertEqual(call_args[0]._output_type, 'table')
    self.assertEqual(call_kwargs['sampling']('test_sql'),
                     bq._sampling.Sampling.default()('test_sql'))

    # test --profile
    args['profile'] = True
    mock_query_execute.return_value.result = lambda: pandas.DataFrame({'c': 'v'}, index=['c'])
    bq.commands._bigquery._sample_cell(args, cell_body)
    call_args = mock_query_execute.call_args[0]
    self.assertEqual(call_args[0]._output_type, 'dataframe')

    # test --verbose
    args['verbose'] = True
    with mock.patch('sys.stdout', new=six.StringIO()) as mocked_stdout:
      bq.commands._bigquery._sample_cell(args, cell_body)
    self.assertEqual(mocked_stdout.getvalue(), 'test_sql\n')
    args['verbose'] = False

    # bad query
    mock_get_notebook_item.return_value = None
    with self.assertRaises(Exception):
      bq.commands._bigquery._sample_cell(args, cell_body)

    # table passed
    args['query'] = None
    args['table'] = 'test.table'
    mock_get_notebook_item.return_value = bq.Table('test.table')
    bq.commands._bigquery._sample_cell(args, cell_body)

    # bad table
    mock_get_table.return_value = None
    with self.assertRaises(Exception):
      bq.commands._bigquery._sample_cell(args, cell_body)

    # view passed
    args['table'] = None
    args['view'] = 'test_view'
    mock_get_notebook_item.return_value = bq.View('test.view')
    bq.commands._bigquery._sample_cell(args, cell_body)

    # bad view
    mock_get_notebook_item.return_value = None
    with self.assertRaises(Exception):
      bq.commands._bigquery._sample_cell(args, cell_body)

  @mock.patch('google.datalab.bigquery.Query.dry_run')
  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.commands._bigquery._get_query_argument')
  def test_dry_run_cell(self, mock_get_query_argument, mock_context_default, mock_dry_run):
    args = {'query': 'test_query'}
    cell_body = ''
    mock_get_query_argument.return_value = bq.Query('test_sql')

    # test --verbose
    args['verbose'] = True
    with mock.patch('sys.stdout', new=six.StringIO()) as mocked_stdout:
      bq.commands._bigquery._dryrun_cell(args, cell_body)
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
    t = bq.commands._bigquery._get_table('bad.name')
    self.assertIsNone(t)

    # test good table name
    test_table_name = 'testproject.test.table'
    mock_get_notebook_item.return_value = bq.Table(test_table_name)
    t = bq.commands._bigquery._get_table(test_table_name)
    self.assertEqual(t.full_name, test_table_name)

    # test table name reference
    mock_get_notebook_item.return_value = test_table_name
    mock_table_exists.return_value = True
    t = bq.commands._bigquery._get_table(test_table_name)
    self.assertEqual(t.full_name, test_table_name)

    self.assertIn(test_table_name, bq.commands._bigquery._existing_table_cache)

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Datasets')
  @mock.patch('google.datalab.bigquery.commands._bigquery._render_list')
  def test_dataset_line_list(self, mock_render_list, mock_datasets, mock_default_context):
    args = {'command': 'list', 'filter': None, 'project': None}
    datasets = ['ds1', 'ds2', 'ds11']
    mock_datasets.return_value = iter(datasets)
    bq.commands._bigquery._dataset_line(args)
    mock_render_list.assert_called_with(datasets)

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Datasets')
  @mock.patch('google.datalab.bigquery.commands._bigquery._render_list')
  def test_dataset_line_list_asterisk(self, mock_render_list, mock_datasets,
                                      mock_default_context):
    args = {'command': 'list', 'filter': '*', 'project': None}
    datasets = ['ds1', 'ds2', 'ds11']
    mock_datasets.return_value = iter(datasets)
    bq.commands._bigquery._dataset_line(args)
    mock_render_list.assert_called_with(datasets)

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Datasets')
  @mock.patch('google.datalab.bigquery.commands._bigquery._render_list')
  def test_dataset_line_list_substr_filter(self, mock_render_list, mock_datasets,
                                           mock_default_context):
    args = {'command': 'list', 'filter': 'ds1*', 'project': None}
    datasets = ['ds1', 'ds2', 'ds11']
    mock_datasets.return_value = iter(datasets)
    bq.commands._bigquery._dataset_line(args)
    mock_render_list.assert_called_with(['ds1', 'ds11'])

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Datasets')
  @mock.patch('google.datalab.bigquery.commands._bigquery._render_list')
  def test_dataset_line_list_exact_filter(self, mock_render_list, mock_datasets,
                                          mock_default_context):
    args = {'command': 'list', 'filter': 'ds1', 'project': None}
    datasets = ['ds1', 'ds2', 'ds11']
    mock_datasets.return_value = iter(datasets)
    bq.commands._bigquery._dataset_line(args)
    mock_render_list.assert_called_with(['ds1'])

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Datasets')
  @mock.patch('google.datalab.bigquery.commands._bigquery._render_list')
  def test_dataset_line_list_project(self, mock_render_list, mock_datasets, mock_default_context):
    args = {'command': 'list', 'filter': None, 'project': 'testproject'}
    mock_default_context.return_value = self._create_context()
    bq.commands._bigquery._dataset_line(args)
    self.assertEqual(mock_datasets.call_args[0][0].project_id, 'testproject')

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Dataset')
  def test_dataset_line_create(self, mock_dataset, mock_default_context):
    args = {'command': 'create', 'name': 'dataset-name', 'friendly': 'test-name'}
    bq.commands._bigquery._dataset_line(args)
    mock_dataset.assert_called_with('dataset-name')
    mock_dataset.return_value.create.assert_called_with(friendly_name='test-name')

    mock_dataset.side_effect = Exception('error')
    with mock.patch('sys.stdout', new=six.StringIO()) as mocked_stdout:
      bq.commands._bigquery._dataset_line(args)
    self.assertIn('Failed to create dataset dataset-name', mocked_stdout.getvalue())

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Dataset')
  def test_dataset_line_delete(self, mock_dataset, mock_default_context):
    args = {'command': 'delete', 'name': 'dataset-name'}
    bq.commands._bigquery._dataset_line(args)
    mock_dataset.assert_called_with('dataset-name')
    mock_dataset.return_value.delete.assert_called_with()

    mock_dataset.side_effect = Exception('error')
    with mock.patch('sys.stdout', new=six.StringIO()) as mocked_stdout:
      bq.commands._bigquery._dataset_line(args)
    self.assertIn('Failed to delete dataset dataset-name', mocked_stdout.getvalue())

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Datasets')
  def test_table_cell_list(self, mock_datasets, mock_default_context):
    args = {'command': 'list', 'filter': None, 'dataset': None, 'project': None}
    tables = [bq.Table('project.test.' + name) for name in ['t1', 't2', 't3']]
    ds1 = mock.MagicMock()
    ds1.__iter__.return_value = iter([tables[0], tables[1]])
    ds2 = mock.MagicMock()
    ds2.__iter__.return_value = iter([tables[2]])
    mock_datasets.return_value = iter([ds1, ds2])
    self.assertEqual(
        bq.commands._bigquery._table_cell(args, None),
        '<ul><li>project.test.t1</li><li>project.test.t2</li><li>project.test.t3</li></ul>')

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Datasets')
  def test_table_cell_list_asterisk(self, mock_datasets, mock_default_context):
    args = {'command': 'list', 'filter': '*', 'dataset': None, 'project': None}
    tables = [bq.Table('project.test.' + name) for name in ['t1', 't2', 't3']]
    ds1 = mock.MagicMock()
    ds1.__iter__.return_value = iter([tables[0], tables[1]])
    ds2 = mock.MagicMock()
    ds2.__iter__.return_value = iter([tables[2]])
    mock_datasets.return_value = iter([ds1, ds2])
    self.assertEqual(
        bq.commands._bigquery._table_cell(args, None),
        '<ul><li>project.test.t1</li><li>project.test.t2</li><li>project.test.t3</li></ul>')

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Datasets')
  def test_table_cell_list_substr_filter(self, mock_datasets, mock_default_context):
    args = {'command': 'list', 'filter': '*t1*', 'dataset': None, 'project': None}
    tables = [bq.Table('project.test.' + name) for name in ['t1', 't2', 't11']]
    ds1 = mock.MagicMock()
    ds1.__iter__.return_value = iter([tables[0], tables[1]])
    ds2 = mock.MagicMock()
    ds2.__iter__.return_value = iter([tables[2]])
    mock_datasets.return_value = iter([ds1, ds2])
    self.assertEqual(
        bq.commands._bigquery._table_cell(args, None),
        '<ul><li>project.test.t1</li><li>project.test.t11</li></ul>')

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Datasets')
  def test_table_cell_list_bad_filter(self, mock_datasets, mock_default_context):
    args = {'command': 'list', 'filter': 't7', 'dataset': None, 'project': None}
    tables = [bq.Table('project.test.' + name) for name in ['t1', 't2', 't11']]
    ds1 = mock.MagicMock()
    ds1.__iter__.return_value = iter([tables[0], tables[1]])
    ds2 = mock.MagicMock()
    ds2.__iter__.return_value = iter([tables[2]])
    mock_datasets.return_value = iter([ds1, ds2])
    self.assertEqual(
        bq.commands._bigquery._table_cell(args, None),
        '<pre>&lt;empty&gt;</pre>')

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Dataset')
  def test_table_cell_list_dataset(self, mock_dataset, mock_default_context):
    args = {'command': 'list', 'filter': '', 'dataset': 'test-dataset', 'project': None}
    tables = [bq.Table('project.test.' + name) for name in ['t1', 't2']]
    mock_dataset.return_value = iter(tables)
    self.assertEqual(
        bq.commands._bigquery._table_cell(args, None),
        '<ul><li>project.test.t1</li><li>project.test.t2</li></ul>')

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Datasets')
  def test_table_cell_list_project(self, mock_datasets, mock_default_context):
    args = {'command': 'list', 'filter': '', 'dataset': None, 'project': 'test-project'}
    tables = [bq.Table('project.test.' + name) for name in ['t1', 't2', 't3']]
    ds1 = mock.MagicMock()
    ds1.__iter__.return_value = iter([tables[0], tables[1]])
    ds2 = mock.MagicMock()
    ds2.__iter__.return_value = iter([tables[2]])
    mock_datasets.return_value = iter([ds1, ds2])
    self.assertEqual(
        bq.commands._bigquery._table_cell(args, None),
        '<ul><li>project.test.t1</li><li>project.test.t2</li><li>project.test.t3</li></ul>')

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Dataset')
  def test_table_cell_list_dataset_project(self, mock_dataset, mock_default_context):
    args = {'command': 'list', 'filter': '', 'dataset': 'test-dataset', 'project': 'test-project'}
    tables = [bq.Table('project.test.' + name) for name in ['t1', 't2']]
    mock_dataset.return_value = iter(tables)
    self.assertEqual(
        bq.commands._bigquery._table_cell(args, None),
        '<ul><li>project.test.t1</li><li>project.test.t2</li></ul>')
    call_args = mock_dataset.call_args[0]
    self.assertEqual(call_args[0], 'test-dataset')
    self.assertEqual(call_args[1].project_id, 'test-project')

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Table')
  def test_table_cell_create_bad_params(self, mock_table, mock_default_context):
    args = {'command': 'create', 'name': 'test-table', 'overwrite': None}
    with mock.patch('sys.stdout', new=six.StringIO()) as mocked_stdout:
      bq.commands._bigquery._table_cell(args, None)
    self.assertIn('Failed to create test-table: no schema', mocked_stdout.getvalue())

    mock_table.side_effect = Exception
    with mock.patch('sys.stdout', new=six.StringIO()) as mocked_stdout:
      bq.commands._bigquery._table_cell(args, json.dumps({}))
    self.assertIn('\'schema\' is a required property', mocked_stdout.getvalue())

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Table')
  def test_table_cell_create(self, mock_table, mock_default_context):
    args = {'command': 'create', 'name': 'test-table', 'overwrite': None}
    cell_body = {
      'schema': [
        {'name': 'col1', 'type': 'int64', 'mode': 'NULLABLE', 'description': 'description1'},
        {'name': 'col1', 'type': 'STRING', 'mode': 'required', 'description': 'description1'}
      ]
    }
    bq.commands._bigquery._table_cell(args, json.dumps(cell_body))
    call_kwargs = mock_table.return_value.create.call_args[1]
    self.assertEqual(None, call_kwargs['overwrite'])
    self.assertEqual(bq.Schema(cell_body['schema']), call_kwargs['schema'])

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.commands._bigquery._get_table')
  def test_table_cell_describe(self, mock_get_table, mock_default_context):
    args = {'command': 'describe', 'name': 'test-table', 'overwrite': None}
    mock_get_table.return_value = None
    with self.assertRaisesRegexp(Exception, 'Could not find table'):
      bq.commands._bigquery._table_cell(args, None)

    mock_get_table.return_value = bq.Table('project.test.table')
    schema = bq.Schema([{
      'name': 'col1',
      'type': 'string'
    }])
    mock_get_table.return_value._schema = schema
    rendered = bq.commands._bigquery._table_cell(args, None)
    expected_html1 = 'bq.renderSchema(dom, [{"type": "string", "name": "col1"}]);'
    expected_html2 = 'bq.renderSchema(dom, [{"name": "col1", "type": "string"}]);'
    self.assertTrue(expected_html1 in rendered or expected_html2 in rendered)

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Table')
  def test_table_cell_delete(self, mock_table, mock_default_context):
    args = {'command': 'delete', 'name': 'test-table'}
    mock_table.return_value.delete.side_effect = Exception
    with mock.patch('sys.stdout', new=six.StringIO()) as mocked_stdout:
      bq.commands._bigquery._table_cell(args, None)
    self.assertIn('Failed to delete table test-table', mocked_stdout.getvalue())

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.commands._bigquery._get_table')
  def test_table_cell_view(self, mock_get_table, mock_default_context):
    args = {'command': 'view', 'name': 'test-table'}
    table = bq.Table('project.test.table')
    mock_get_table.return_value = None
    with self.assertRaisesRegexp(Exception, 'Could not find table test-table'):
      bq.commands._bigquery._table_cell(args, None)

    mock_get_table.return_value = table
    self.assertEqual(table, bq.commands._bigquery._table_cell(args, None))

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Query.execute')
  @mock.patch('google.datalab.utils.commands.get_notebook_item')
  def test_extract_cell_query(self, mock_get_notebook_item, mock_query_execute,
                              mock_default_context):
    args = {'table': None, 'view': None, 'query': None, 'path': None, 'format': None,
            'delimiter': None, 'header': None, 'compress': None, 'nocache': None}
    with self.assertRaisesRegexp(Exception, 'A query, table, or view is needed'):
      bq.commands._bigquery._extract_cell(args, None)

    args['query'] = 'test-query'
    mock_get_notebook_item.return_value = None
    with self.assertRaisesRegexp(Exception, 'Could not find query test-query'):
      bq.commands._bigquery._extract_cell(args, None)

    mock_get_notebook_item.return_value = bq.Query('sql')
    mock_query_execute.return_value.failed = True
    mock_query_execute.return_value.fatal_error = 'test-error'
    with self.assertRaisesRegexp(Exception, 'Extract failed: test-error'):
      bq.commands._bigquery._extract_cell(args, None)

    mock_query_execute.return_value.failed = False
    mock_query_execute.return_value.errors = 'test-errors'
    with self.assertRaisesRegexp(Exception, 'Extract completed with errors: test-errors'):
      bq.commands._bigquery._extract_cell(args, None)

    mock_query_execute.return_value.errors = None
    mock_query_execute.return_value.result = lambda: 'results'
    self.assertEqual(bq.commands._bigquery._extract_cell(args, None),
                     'results')

    cell_body = {
      'parameters': [
          {'name': 'arg1', 'type': 'INT64', 'value': 5}
      ]
    }
    bq.commands._bigquery._extract_cell(args, json.dumps(cell_body))
    mock_get_notebook_item.assert_called_with('test-query')
    call_args = mock_query_execute.call_args[1]

    found_item = False
    for item in call_args['query_params']:
      if item['name'] == 'arg1':
        found_item = True
        self.assertDictEqual(item, {
          'parameterValue': {'value': 5},
          'name': 'arg1',
          'parameterType': {'type': 'INT64'}
        })
    self.assertTrue(found_item)

  @mock.patch('google.datalab.bigquery.Table.extract')
  @mock.patch('google.datalab.bigquery.commands._bigquery._get_table')
  @mock.patch('google.datalab.utils.commands.get_notebook_item')
  def test_extract_cell_table(self, mock_get_notebook_item, mock_get_table, mock_table_extract):
    args = {'table': 'test-table', 'path': 'test-path', 'format': 'json', 'delimiter': None,
            'header': None, 'compress': None, 'nocache': None}
    mock_get_table.return_value = None
    with self.assertRaisesRegexp(Exception, 'Could not find table test-table'):
      bq.commands._bigquery._extract_cell(args, None)

    mock_get_table.return_value = bq.Table('project.test.table', self._create_context())
    mock_table_extract.return_value.result = lambda: 'test-results'
    mock_table_extract.return_value.failed = False
    mock_table_extract.return_value.errors = None
    self.assertEqual(bq.commands._bigquery._extract_cell(args, None),
                     'test-results')
    mock_table_extract.assert_called_with('test-path', format='json', csv_delimiter=None,
                                          csv_header=None, compress=None)

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Query.execute')
  @mock.patch('google.datalab.utils.commands.get_notebook_item')
  def test_extract_cell_view(self, mock_get_notebook_item, mock_query_execute,
                             mock_default_context):
    args = {'view': 'test-view', 'table': None, 'query': None, 'path': 'test-path',
            'format': None, 'delimiter': None, 'header': None, 'compress': None, 'nocache': None}
    mock_get_notebook_item.return_value = None
    with self.assertRaisesRegexp(Exception, 'Could not find view test-view'):
      bq.commands._bigquery._extract_cell(args, None)

    mock_get_notebook_item.return_value = bq.View('project.test.view', self._create_context())
    mock_query_execute.return_value.result = lambda: 'test-results'
    mock_query_execute.return_value.failed = False
    mock_query_execute.return_value.errors = None
    self.assertEqual(bq.commands._bigquery._extract_cell(args, None),
                     'test-results')
    mock_get_notebook_item.assert_called_with('test-view')

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.bigquery.Table.create')
  @mock.patch('google.datalab.bigquery.Table.exists')
  @mock.patch('google.datalab.bigquery.Table.load')
  @mock.patch('google.datalab.bigquery.commands._bigquery._get_table')
  def test_load_cell(self, mock_get_table, mock_table_load, mock_table_exists,
                     mock_table_create, mock_default_context):

    args = {'table': 'project.test.table', 'mode': 'create', 'path': 'test/path_%(_ds)s',
            'skip': None, 'csv': None, 'delimiter': None, 'format': 'csv', 'strict': None,
            'quote': None}
    context = self._create_context()
    mock_get_table.return_value = bq.Table('project.test.table')
    job = bq._query_job.QueryJob('test_id', 'project.test.table', 'test_sql', context)

    mock_table_exists.return_value = True
    with self.assertRaisesRegexp(Exception, 'already exists; use "append" or "overwrite" as mode.'):
      bq.commands._bigquery._load_cell(args, None)

    mock_table_exists.return_value = False
    with self.assertRaisesRegexp(Exception, 'Table does not exist, and no schema specified'):
      bq.commands._bigquery._load_cell(args, None)

    cell_body = {
      'schema': [
        {'name': 'col1', 'type': 'int64', 'mode': 'NULLABLE', 'description': 'description1'},
        {'name': 'col1', 'type': 'STRING', 'mode': 'required', 'description': 'description1'}
      ],
      'parameters': [
        {'name': 'custom', 'type': 'FLOAT', 'value': 4.23}
      ]
    }

    mock_table_load.return_value = job
    job._is_complete = True
    job._fatal_error = 'fatal error'
    with self.assertRaisesRegexp(Exception, 'Load failed: fatal error'):
      bq.commands._bigquery._load_cell(args, json.dumps(cell_body))

    job._fatal_error = None
    job._errors = 'error'
    with self.assertRaisesRegexp(Exception, 'Load completed with errors: error'):
      bq.commands._bigquery._load_cell(args, json.dumps(cell_body))

    job._errors = None
    bq.commands._bigquery._load_cell(args, json.dumps(cell_body))
    today = datetime.now().date().isoformat()
    mock_table_load.assert_called_with('test/path_{0}'.format(today), mode='create',
                                       source_format='csv',
                                       csv_options=mock.ANY, ignore_unknown_values=True)

    mock_get_table.return_value = None
    mock_table_exists.return_value = True
    args['mode'] = 'append'
    args['format'] = 'csv'

    bq.commands._bigquery._load_cell(args, None)
    mock_table_load.assert_called_with('test/path_{0}'.format(today), mode='append',
                                       source_format='csv', csv_options=mock.ANY,
                                       ignore_unknown_values=True)

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.cloud.storage.Client')
  @mock.patch('google.cloud.storage.Blob')
  @mock.patch('google.cloud.storage.Client.get_bucket')
  @mock.patch('google.datalab.utils.commands.get_notebook_item')
  @mock.patch('google.datalab.utils.commands.notebook_environment')
  def test_pipeline_cell(self, mock_env, mock_get_notebook_item, mock_client_get_bucket,
                         mock_blob_class, mock_client, mock_default_context):
    context = TestCases._create_context()
    mock_default_context.return_value = context
    mock_client_get_bucket.return_value = mock.Mock(spec=google.cloud.storage.Bucket)
    mock_get_notebook_item.return_value = google.datalab.bigquery.Query(
        'SELECT * FROM publicdata.samples.wikipedia LIMIT 5')
    args = {'name': 'bq_pipeline_test'}
    small_cell_body = """
            emails: foo1@test.com
            schedule:
                start: 2009-05-05T22:28:15Z
                end: 2009-05-06T22:28:15Z
                interval: '@hourly'
            input:
                table: project.test.table
            transformation:
                query: foo_query
            output:
                table: project.test.table
       """

    actual = google.datalab.contrib.bigquery.commands._bigquery._pipeline_cell(args,
                                                                               small_cell_body)
    self.assertIn("'email': ['foo1@test.com']", actual)

  @mock.patch('google.datalab.utils.commands._html.Html.next_id')
  @mock.patch('google.datalab.utils.commands._html.HtmlBuilder.render_chart_data')
  @mock.patch('google.datalab.bigquery._api.Api.tables_get')
  @mock.patch('google.datalab.utils.commands.get_data')
  @mock.patch('google.datalab.utils.commands.get_field_list')
  @mock.patch('google.datalab.bigquery.Table.exists')
  def test_table_viewer(self, mock_table_exists, mock_get_field_list, mock_get_data,
                        mock_tables_get, mock_render_chart_data, mock_next_id):
    test_table = bq.Table('testproject.test.table', self._create_context())

    mock_table_exists.return_value = False
    with self.assertRaisesRegexp(Exception, 'does not exist'):
      bq.commands._bigquery._table_viewer(test_table)

    mock_table_exists.return_value = True
    mock_get_field_list.return_value = ['col1']
    mock_get_data.return_value = ({'cols': ['col1'], 'rows': ['val1']}, 1)
    mock_render_chart_data.return_value = 'test_chart_data'
    mock_next_id.return_value = 'test_id'
    viewer = bq.commands._bigquery._table_viewer(test_table)

    mock_table_exists.assert_called()
    mock_get_field_list.assert_called()
    mock_render_chart_data.assert_called()

    expected_html_header = '''
    <div class="bqtv" id="test_id">test_chart_data</div>
    <br />(testproject.test.table)<br />
    '''
    self.assertIn(expected_html_header, viewer)

  @mock.patch('google.datalab.bigquery._query_stats.QueryStats._size_formatter')
  @mock.patch('google.datalab.bigquery.Table.job')
  @mock.patch('google.datalab.utils.commands._html.Html.next_id')
  @mock.patch('google.datalab.utils.commands._html.HtmlBuilder.render_chart_data')
  @mock.patch('google.datalab.bigquery._api.Api.tables_get')
  @mock.patch('google.datalab.utils.commands.get_data')
  @mock.patch('google.datalab.utils.commands.get_field_list')
  @mock.patch('google.datalab.bigquery.Table.exists')
  def test_query_results_table_viewer(self, mock_table_exists, mock_get_field_list, mock_get_data,
                                      mock_tables_get, mock_render_chart_data, mock_next_id,
                                      mock_table_job, mock_size_formatter):
    context = self._create_context()
    table_name = 'testproject.test.table'
    job = bq._query_job.QueryJob('test_id', table_name, 'test_sql', context)
    job._start_time, job._end_time = datetime(2017, 1, 1, 1, 1), datetime(2017, 1, 1, 1, 2)
    test_table = bq.QueryResultsTable(table_name, context, job)

    mock_table_exists.return_value = True
    mock_get_field_list.return_value = ['col1']
    mock_get_data.return_value = ({'cols': ['col1'], 'rows': ['val1']}, 1)
    mock_next_id.return_value = 'test_id'
    mock_size_formatter.return_value = '10MB'
    mock_render_chart_data.return_value = 'test_chart_data'

    viewer = bq.commands._bigquery._table_viewer(test_table)

    mock_table_exists.assert_called()
    mock_get_field_list.assert_called()
    mock_render_chart_data.assert_called()

    expected_html_header = '''
    <div class="bqtv" id="test_id">test_chart_data</div>
    <br />(time: 60.0s, 10MB processed, job: test_id)<br />
    '''
    self.assertIn(expected_html_header, viewer)

    job._cache_hit = True

    viewer = bq.commands._bigquery._table_viewer(test_table)

    expected_html_header = '''
    <div class="bqtv" id="test_id">test_chart_data</div>
    <br />(time: 60.0s, cached, job: test_id)<br />
    '''
    self.assertIn(expected_html_header, viewer)

    mock_get_data.return_value = ({'rows': []}, -1)
    viewer = bq.commands._bigquery._table_viewer(test_table)
    expected_html_header = 'pageSize: 25,'
    self.assertIn(expected_html_header, viewer)

    mock_get_data.return_value = ({'rows': ['val'] * 5}, -1)
    viewer = bq.commands._bigquery._table_viewer(test_table, rows_per_page=10)
    expected_html_header = 'pageSize: 10,'
    self.assertIn(expected_html_header, viewer)
    expected_html_footer = '''
            {source_index: 0, fields: 'col1'},
            0,
            5);
    '''
    self.assertIn(expected_html_footer, viewer)

  @mock.patch('google.datalab.utils._utils.get_credentials')
  @mock.patch('google.datalab.utils._utils.get_default_project_id')
  @mock.patch('google.datalab.utils._utils.save_project_id')
  def test_args_to_context(self, mock_save_project, mock_get_default_project, mock_get_credentials):
    mock_get_credentials.return_value = 'test_creds'
    mock_get_default_project.return_value = 'testproject'

    args = {'billing': 'billing_value'}
    default_context = google.datalab.Context.default()
    c = google.datalab.utils._utils._construct_context_for_args(args)

    # make sure it's not the same object
    self.assertNotEqual(c, default_context)
    self.assertEqual(c.project_id, default_context.project_id)
    self.assertEqual(c.credentials, default_context.credentials)

    # make sure the right config object was passed
    self.assertEqual(c.config, {'bigquery_billing_tier': 'billing_value'})

    default_context.config['test_prop'] = 'test_val'
    c = google.datalab.utils._utils._construct_context_for_args(args)
    # make sure other properties in default context were copied
    self.assertEqual(c.config, {'bigquery_billing_tier': 'billing_value', 'test_prop': 'test_val'})

  @mock.patch('google.datalab.utils.commands.get_notebook_item')
  def test_get_query_argument(self, mock_get_notebook_item):
    args = {}
    cell = None
    env = {}
    # an Exception should be raised if no query is specified by name or body
    with self.assertRaises(Exception):
      bq.commands._bigquery._get_query_argument(args, cell, env)

    # specify query name, no cell body
    args = {'query': 'test_query'}
    mock_get_notebook_item.return_value = bq.Query('test_sql')
    q = bq.commands._bigquery._get_query_argument(args, cell, env)
    self.assertEqual(q.sql, 'test_sql')

    # specify query in cell body, no name
    args = {}
    cell = 'test_sql2'
    q = bq.commands._bigquery._get_query_argument(args, cell, env)
    self.assertEqual(q.sql, 'test_sql2')

    # specify query by bad name
    args = {'query': 'test_query'}
    mock_get_notebook_item.return_value = None
    with self.assertRaises(Exception):
      bq.commands._bigquery._get_query_argument(args, cell, env)

  def test_get_query_parameters(self):
    args = {'query': None}
    cell_body = ''
    now = datetime.now()
    with self.assertRaises(Exception):
      bq.commands._bigquery.get_query_parameters(args, json.dumps(cell_body))

    args['query'] = 'test_sql'
    params = bq.commands._bigquery.get_query_parameters(args, json.dumps(cell_body), date_time=now)

    # We push the params into a dict so that it's easier to compare
    params_dict = {
      item['name']: {
        'type': item['parameterType']['type'],
        'value': item['parameterValue']['value']
      } for item in params
    }

    today = now.date()
    default_query_parameters = {
      # the datetime formatted as YYYY-MM-DD
      '_ds': {'type': 'STRING', 'value': today.isoformat()},
      # the full ISO-formatted timestamp YYYY-MM-DDTHH:MM:SS.mmmmmm
      '_ts': {'type': 'STRING', 'value': now.isoformat()},
      # the datetime formatted as YYYYMMDD (i.e. YYYY-MM-DD with 'no dashes')
      '_ds_nodash': {'type': 'STRING', 'value': today.strftime('%Y%m%d')},
      # the timestamp formatted as YYYYMMDDTHHMMSSmmmmmm (i.e full ISO-formatted timestamp
      # YYYY-MM-DDTHH:MM:SS.mmmmmm with no dashes or colons).
      '_ts_nodash': {'type': 'STRING', 'value': now.strftime('%Y%m%d%H%M%S%f')},
      '_ts_year': {'type': 'STRING', 'value': today.strftime('%Y')},
      '_ts_month': {'type': 'STRING', 'value': today.strftime('%m')},
      '_ts_day': {'type': 'STRING', 'value': today.strftime('%d')},
      '_ts_hour': {'type': 'STRING', 'value': now.strftime('%H')},
      '_ts_minute': {'type': 'STRING', 'value': now.strftime('%M')},
      '_ts_second': {'type': 'STRING', 'value': now.strftime('%S')},
    }
    self.assertDictEqual(params_dict, default_query_parameters)

    cell_body = {
      'parameters': [
          {'name': 'arg1', 'type': 'INT64', 'value': 5},
          {'name': 'arg2', 'type': 'string', 'value': 'val2'},
          {'name': 'arg3', 'type': 'date', 'value': 'val3'}
      ]
    }
    params = bq.commands._bigquery.get_query_parameters(args, json.dumps(cell_body), date_time=now)
    # We push the params into a dict so that it's easier to compare
    params_dict = {
      item['name']: {
        'type': item['parameterType']['type'],
        'value': item['parameterValue']['value']
      } for item in params
    }
    cell_body_params_dict = {
      item['name']: {
        'type': item['type'],
        'value': item['value']
      } for item in cell_body['parameters']
    }
    default_query_parameters.update(cell_body_params_dict)
    self.assertDictEqual(params_dict, default_query_parameters)
