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
import google.datalab.contrib.pipeline._pipeline as pipeline
import mock
import pickle
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

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = mock.Mock(spec=google.auth.credentials.Credentials)
    return google.datalab.Context(project_id, creds)

  @mock.patch('google.datalab.bigquery.Query.execute')
  @mock.patch('google.datalab.utils.commands.get_notebook_item')
  def test_execute_operator_definition(self, mock_get_notebook_item, mock_query_execute):
    mock_get_notebook_item.return_value = google.datalab.bigquery.Query('test_sql')
    task_id = 'foo'
    task_details = {}
    task_details['type'] = 'pydatalab.bq.execute'
    task_details['query'] = 'test_sql'
    context = TestCases._create_context()
    task_details['py_context'] = pickle.dumps(context, -1)
    task_details['mode'] = 'create'

    operator_def = pipeline.Pipeline(None, None)._get_operator_definition(
        task_id, task_details)
    self.assertEqual(operator_def, """foo = ExecuteOperator(task_id=\'foo_id\', mode=\'create\', py_context=\'\x80\x02cgoogle.datalab._context\nContext\nq\x00)\x81q\x01}q\x02(U\x0b_project_idq\x03X\x04\x00\x00\x00testq\x04U\x07_configq\x05}q\x06X\x15\x00\x00\x00bigquery_billing_tierq\x07NsU\x0c_credentialsq\x08cgoogle.auth.credentials\nCredentials\nq\t)\x81q\n}q\x0b(U\x0cmethod_callsq\x0ccmock.mock\n_CallList\nq\r)\x81q\x0e}q\x0fbU\x0e_mock_new_nameq\x10U\x00q\x11U\r_mock_methodsq\x12]q\x13(U\x13__abstractmethods__q\x14U\t__class__q\x15U\x0b__delattr__q\x16U\x08__dict__q\x17U\x07__doc__q\x18U\n__format__q\x19U\x10__getattribute__q\x1aU\x08__hash__q\x1bU\x08__init__q\x1cU\n__module__q\x1dU\x07__new__q\x1eU\n__reduce__q\x1fU\r__reduce_ex__q U\x08__repr__q!U\x0b__setattr__q"U\n__sizeof__q#U\x07__str__q$U\x10__subclasshook__q%U\x0b__weakref__q&U\n_abc_cacheq\'U\x13_abc_negative_cacheq(U\x1b_abc_negative_cache_versionq)U\r_abc_registryq*U\x05applyq+U\x0ebefore_requestq,U\x07expiredq-U\x07refreshq.U\x05validq/eU\x0c_mock_parentq0NU\x0c_mock_unsafeq1\x89U\x0f_mock_call_argsq2NU\x0e_mock_childrenq3}q4U\x0f_spec_signatureq5cfuncsigs\nSignature\nq6)\x81q7N}q8(U\x0b_parametersq9ccollections\nOrderedDict\nq:]q;\x85q<Rq=U\x12_return_annotationq>cfuncsigs\n_empty\nq?u\x86q@bU\x0b_mock_wrapsqANU\x10_mock_call_countqBK\x00U\x11_mock_side_effectqCNU\x0b_spec_classqDh\tU\x0c_mock_calledqE\x89U\x14_mock_call_args_listqFh\r)\x81qG}qHbU\x10_mock_mock_callsqIh\r)\x81qJ}qKbU\t_spec_setqLNU\x0e_mock_delegateqMNU\x10_mock_new_parentqNNU\x12_mock_return_valueqOcmock.mock\n_SentinelObject\nqP)\x81qQ}qRU\x04nameqSU\x07DEFAULTqTsbU\n_mock_nameqUNubub.\', query=\'test_sql\', dag=dag)\n""")  # noqa

  @mock.patch('google.datalab.bigquery.Query.execute')
  @mock.patch('google.datalab.utils.commands.get_notebook_item')
  def test_execute_operator(self, mock_get_notebook_item, mock_query_execute):
    mock_get_notebook_item.return_value = google.datalab.bigquery.Query('test_sql')
    context = self._create_context()
    # This statement is required even though it seems like it's not. Go figure.
    execute_operator = ExecuteOperator(
      task_id='test_execute_operator', query='test_sql', parameters=None, table='test_table',
      mode=None, py_context_str=pickle.dumps(context))
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
