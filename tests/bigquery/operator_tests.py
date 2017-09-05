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

from google.datalab.contrib.bigquery.operators.bq_extract_operator import ExtractOperator  # noqa
from google.datalab.contrib.bigquery.operators.bq_execute_operator import ExecuteOperator  # noqa
from google.datalab.contrib.bigquery.operators.bq_load_operator import LoadOperator  # noqa


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
    extract_operator = ExtractOperator(task_id='test_extract_operator', table='test_table',
                                       path='test_path', format=None, delimiter=None, header=None,
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
  @mock.patch('google.datalab.utils.commands.get_notebook_item')
  def test_execute_operator(self, mock_get_notebook_item, mock_query_execute):
    mock_get_notebook_item.return_value = google.datalab.bigquery.Query('test_sql')
    execute_operator = ExecuteOperator(
      task_id='test_execute_operator', query='test_sql', parameters=None, table='test_table',
      mode=None, billing='test_billing')
    execute_operator.execute(context=None)
    # TODO(rajivpb): Mock output_options, context, and query_params for a more complete test.
    mock_query_execute.assert_called_once()

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
