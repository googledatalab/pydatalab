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
from builtins import str
import datetime
import mock
import unittest

import google.auth
import google.datalab
import google.datalab.bigquery


class TestCases(unittest.TestCase):

  def test_parameter_validation(self):
    sql = 'SELECT * FROM table'
    with self.assertRaises(Exception):
      TestCases._create_query(sql, subqueries=['subquery'])
    sq = TestCases._create_query()
    env = {'subquery': sq}
    q = TestCases._create_query(sql, env=env, subqueries=['subquery'])
    self.assertIsNotNone(q)
    self.assertEqual(q.subqueries, {'subquery': sq})
    self.assertEqual(q._sql, sql)

    with self.assertRaises(Exception):
      TestCases._create_query(sql, udfs=['udf'])
    udf = TestCases._create_udf('test_udf', 'code', 'TYPE')
    env = {'testudf': udf}
    q = TestCases._create_query(sql, env=env, udfs=['testudf'])
    self.assertIsNotNone(q)
    self.assertEqual(q.udfs, {'testudf': udf})
    self.assertEqual(q._sql, sql)

    with self.assertRaises(Exception):
      TestCases._create_query(sql, data_sources=['test_datasource'])
    test_datasource = TestCases._create_data_source('gs://test/path')
    env = {'test_datasource': test_datasource}
    q = TestCases._create_query(sql, env=env, data_sources=['test_datasource'])
    self.assertIsNotNone(q)
    self.assertEqual(q.data_sources, {'test_datasource': test_datasource})
    self.assertEqual(q._sql, sql)

  def test_query_with_udf_object(self):
    udf = TestCases._create_udf('test_udf', 'udf body', 'TYPE')
    q = TestCases._create_query('SELECT * FROM table', udfs={'test_udf': udf})
    self.assertIn('udf body', q.sql)

  @mock.patch('google.datalab.bigquery._api.Api.tabledata_list')
  @mock.patch('google.datalab.bigquery._api.Api.jobs_insert_query')
  @mock.patch('google.datalab.bigquery._api.Api.jobs_query_results')
  @mock.patch('google.datalab.bigquery._api.Api.jobs_get')
  @mock.patch('google.datalab.bigquery._api.Api.tables_get')
  def test_single_result_query(self, mock_api_tables_get, mock_api_jobs_get,
                               mock_api_jobs_query_results, mock_api_insert_query,
                               mock_api_tabledata_list):
    mock_api_tables_get.return_value = TestCases._create_tables_get_result()
    mock_api_jobs_get.return_value = {'status': {'state': 'DONE'}}
    mock_api_jobs_query_results.return_value = {'jobComplete': True}
    mock_api_insert_query.return_value = TestCases._create_insert_done_result()
    mock_api_tabledata_list.return_value = TestCases._create_single_row_result()

    sql = 'SELECT field1 FROM [table] LIMIT 1'
    q = TestCases._create_query(sql)
    context = TestCases._create_context()
    results = q.execute(context=context).result()

    self.assertEqual(sql, results.sql)
    self.assertEqual('(%s)' % sql, q._repr_sql_())
    self.assertEqual(1, results.length)
    first_result = results[0]
    self.assertEqual('value1', first_result['field1'])

  @mock.patch('google.datalab.bigquery._api.Api.jobs_insert_query')
  @mock.patch('google.datalab.bigquery._api.Api.jobs_query_results')
  @mock.patch('google.datalab.bigquery._api.Api.jobs_get')
  @mock.patch('google.datalab.bigquery._api.Api.tables_get')
  def test_empty_result_query(self, mock_api_tables_get, mock_api_jobs_get,
                              mock_api_jobs_query_results, mock_api_insert_query):
    mock_api_tables_get.return_value = TestCases._create_tables_get_result(0)
    mock_api_jobs_get.return_value = {'status': {'state': 'DONE'}}
    mock_api_jobs_query_results.return_value = {'jobComplete': True}
    mock_api_insert_query.return_value = TestCases._create_insert_done_result()

    q = TestCases._create_query()
    context = TestCases._create_context()
    results = q.execute(context=context).result()

    self.assertEqual(0, results.length)

  @mock.patch('google.datalab.bigquery._api.Api.jobs_insert_query')
  @mock.patch('google.datalab.bigquery._api.Api.jobs_query_results')
  @mock.patch('google.datalab.bigquery._api.Api.jobs_get')
  @mock.patch('google.datalab.bigquery._api.Api.tables_get')
  def test_incomplete_result_query(self,
                                   mock_api_tables_get,
                                   mock_api_jobs_get,
                                   mock_api_jobs_query_results,
                                   mock_api_insert_query):
    mock_api_tables_get.return_value = TestCases._create_tables_get_result()
    mock_api_jobs_get.return_value = {'status': {'state': 'DONE'}}
    mock_api_jobs_query_results.return_value = {'jobComplete': True}
    mock_api_insert_query.return_value = TestCases._create_incomplete_result()

    q = TestCases._create_query()
    context = TestCases._create_context()
    results = q.execute(context=context).result()

    self.assertEqual(1, results.length)
    self.assertEqual('test_job', results.job_id)

  @mock.patch('google.datalab.bigquery._api.Api.jobs_insert_query')
  def test_malformed_response_raises_exception(self, mock_api_insert_query):
    mock_api_insert_query.return_value = {}

    q = TestCases._create_query()

    with self.assertRaises(Exception) as error:
      context = TestCases._create_context()
      q.execute(context=context).result()
    self.assertEqual('Unexpected response from server', str(error.exception))

  def test_nested_subquery_expansion(self):
    # test expanding subquery and udf validation
    with self.assertRaises(Exception):
      TestCases._create_query('SELECT * FROM subquery', subqueries=['subquery'])

    with self.assertRaises(Exception):
      TestCases._create_query('SELECT test_udf(field1) FROM test_table', udfs=['test_udf'])

    env = {}

    # test direct subquery expansion
    q1 = TestCases._create_query('SELECT * FROM test_table', name='q1', env=env)
    q2 = TestCases._create_query('SELECT * FROM q1', name='q2', subqueries=['q1'], env=env)
    self.assertEqual('''\
WITH q1 AS (
  SELECT * FROM test_table
)

SELECT * FROM q1''', q2.sql)

    # test recursive, second level subquery expansion
    q3 = TestCases._create_query('SELECT * FROM q2', name='q3', subqueries=['q2'], env=env)
    # subquery listing order is random, try both possibilities
    expected_sql1 = '''\
WITH q1 AS (
  %s
),
q2 AS (
  %s
)

%s''' % (q1._sql, q2._sql, q3._sql)
    expected_sql2 = '''\
WITH q2 AS (
  %s
),
q1 AS (
  %s
)

%s''' % (q2._sql, q1._sql, q3._sql)

    self.assertTrue((expected_sql1 == q3.sql) or (expected_sql2 == q3.sql))

  # @mock.patch('google.datalab.bigquery._api.Api.jobs_insert_query')
  def test_subquery_expansion_order(self):
    env = {}
    TestCases._create_query('SELECT * FROM test_table', name='snps', env=env)
    TestCases._create_query('SELECT * FROM snps', subqueries=['snps'], name='windows', env=env)
    titv = TestCases._create_query('SELECT * FROM snps, windows', subqueries=['snps', 'windows'],
                                   env=env)

    # make sure snps appears before windows in the expanded sql of titv
    snps_pos, windows_pos = titv.sql.find('snps AS'), titv.sql.find('windows AS')
    self.assertNotEqual(snps_pos, -1, 'Could not find snps definition in expanded sql')
    self.assertNotEqual(windows_pos, -1, 'Could not find windows definition in expanded sql')
    self.assertLess(snps_pos, windows_pos)

    # reverse the order they're referenced in titv, and make sure snps still appears before windows
    titv = TestCases._create_query('SELECT * FROM snps, windows', subqueries=['windows', 'snps'],
                                   env=env)
    snps_pos, windows_pos = titv.sql.find('snps AS'), titv.sql.find('windows AS')
    self.assertNotEqual(snps_pos, -1, 'Could not find snps definition in expanded sql')
    self.assertNotEqual(windows_pos, -1, 'Could not find windows definition in expanded sql')
    self.assertLess(snps_pos, windows_pos)

  @staticmethod
  def _create_query(sql='SELECT * ...', name=None, env=None, udfs=None, data_sources=None,
                    subqueries=None):
    if env is None:
      env = {}
    q = google.datalab.bigquery.Query(sql, env=env, udfs=udfs, data_sources=data_sources,
                                      subqueries=subqueries)
    if name:
      env[name] = q
    return q

  @staticmethod
  def _create_udf(name, code, return_type):
    return google.datalab.bigquery.UDF(name, code, return_type)

  @staticmethod
  def _create_data_source(source):
    return google.datalab.bigquery.ExternalDataSource(source=source)

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = mock.Mock(spec=google.auth.credentials.Credentials)
    return google.datalab.Context(project_id, creds)

  @staticmethod
  def _create_insert_done_result():
    # pylint: disable=g-continuation-in-parens-misaligned
    return {
      'jobReference': {
        'jobId': 'test_job'
      },
      'configuration': {
        'query': {
          'destinationTable': {
            'projectId': 'project',
            'datasetId': 'dataset',
            'tableId': 'table'
          }
        }
      },
      'jobComplete': True,
    }

  @staticmethod
  def _create_single_row_result():
    # pylint: disable=g-continuation-in-parens-misaligned
    return {
      'totalRows': 1,
      'rows': [
        {'f': [{'v': 'value1'}]}
      ]
    }

  @staticmethod
  def _create_empty_result():
    # pylint: disable=g-continuation-in-parens-misaligned
    return {
      'totalRows': 0
    }

  @staticmethod
  def _create_incomplete_result():
    # pylint: disable=g-continuation-in-parens-misaligned
    return {
      'jobReference': {
        'jobId': 'test_job'
      },
      'configuration': {
        'query': {
          'destinationTable': {
            'projectId': 'project',
            'datasetId': 'dataset',
            'tableId': 'table'
          }
        }
      },
      'jobComplete': False
    }

  @staticmethod
  def _create_page_result(page_token=None):
    # pylint: disable=g-continuation-in-parens-misaligned
    return {
      'totalRows': 2,
      'rows': [
        {'f': [{'v': 'value1'}]}
      ],
      'pageToken': page_token
    }

  @staticmethod
  def _create_tables_get_result(num_rows=1, schema=None):
    if schema is None:
      schema = [{'name': 'field1', 'type': 'string'}]
    return {
      'numRows': num_rows,
      'schema': {
        'fields': schema
      },
    }

  def test_merged_parameters(self):
    parameters = [
        {'type': 'foo1', 'name': 'foo1', 'value': 'foo1'},
        {'type': 'foo2', 'name': 'foo2', 'value': 'foo2'},
        {'type': 'foo3', 'name': '_ds', 'value': 'foo3'},
    ]
    merged_parameters = google.datalab.bigquery.Query.merge_parameters(
      parameters, date_time=datetime.datetime.now(), macros=True, types_and_values=False)
    expected = {
      'foo1': 'foo1',
      'foo2': 'foo2',
      '_ds': 'foo3',
      '_ts': '{{ ts }}',
      '_ds_nodash': '{{ ds_nodash }}',
      '_ts_nodash': '{{ ts_nodash }}',
      '_ts_year': '{{ execution_date.year }}',
      '_ts_month': '{{ execution_date.month }}',
      '_ts_day': '{{ execution_date.day }}',
      '_ts_hour': '{{ execution_date.hour }}',
      '_ts_minute': '{{ execution_date.minute }}',
      '_ts_second': '{{ execution_date.second }}',
    }
    self.assertDictEqual(merged_parameters, expected)

    date_time = datetime.datetime.now()
    day = date_time.date()

    merged_parameters = google.datalab.bigquery.Query.merge_parameters(
      parameters, date_time=date_time, macros=False, types_and_values=False)
    expected = {
      u'_ts_nodash': date_time.strftime('%Y%m%d%H%M%S%f'),
      u'_ts_second': date_time.strftime('%S'),
      u'_ts_day': day.strftime('%d'),
      u'_ts_minute': date_time.strftime('%M'),
      u'_ts': date_time.isoformat(),
      u'_ts_hour': date_time.strftime('%H'),
      u'_ts_month': day.strftime('%m'),
      u'_ds_nodash': day.strftime('%Y%m%d'),
      u'_ds': 'foo3',
      u'_ts_year': day.strftime('%Y'),
      u'foo1': 'foo1',
      u'foo2': 'foo2'
    }
    self.assertDictEqual(merged_parameters, expected)

    merged_parameters = google.datalab.bigquery.Query.merge_parameters(
      parameters, date_time=date_time, macros=False, types_and_values=True)
    expected = {
      u'_ts_nodash': {u'type': u'STRING', u'value': date_time.strftime('%Y%m%d%H%M%S%f')},
      u'_ts_second': {u'type': u'STRING', u'value': date_time.strftime('%S')},
      u'_ts_day': {u'type': u'STRING', u'value': day.strftime('%d')},
      u'_ts_minute': {u'type': u'STRING', u'value': date_time.strftime('%M')},
      u'_ts': {u'type': u'STRING', u'value': date_time.isoformat()},
      u'_ts_hour': {u'type': u'STRING', u'value': date_time.strftime('%H')},
      u'_ts_month': {u'type': u'STRING', u'value': day.strftime('%m')},
      u'_ds_nodash': {u'type': u'STRING', u'value': day.strftime('%Y%m%d')},
      u'_ds': {u'type': u'foo3', u'value': 'foo3'},
      u'_ts_year': {u'type': u'STRING', u'value': day.strftime('%Y')},
      u'foo1': {u'type': u'foo1', u'value': u'foo1'},
      u'foo2': {u'type': u'foo2', u'value': u'foo2'}
    }
    self.assertDictEqual(merged_parameters, expected)

  def test_resolve_parameters(self):
    date_time = datetime.datetime.now()
    day = date_time.date()
    day_string = day.isoformat()
    self.assertEqual(google.datalab.bigquery.Query.resolve_parameters('foo%(_ds)s', []),
                     'foo{0}'.format(day_string))
    self.assertListEqual(google.datalab.bigquery.Query.resolve_parameters(
      ['foo%(_ds)s', 'bar%(_ds)s'], []), ['foo{0}'.format(day_string), 'bar{0}'.format(day_string)])
    self.assertDictEqual(google.datalab.bigquery.Query.resolve_parameters(
      {'key%(_ds)s': 'value%(_ds)s'}, []),
      {'key{0}'.format(day_string): 'value{0}'.format(day_string)})
    self.assertDictEqual(google.datalab.bigquery.Query.resolve_parameters(
      {'key%(_ds)s': {'key': 'value%(_ds)s'}}, []),
      {'key{0}'.format(day_string): {'key': 'value{0}'.format(day_string)}})
    params = [{'name': 'custom_key', 'value': 'custom_value'}]
    self.assertDictEqual(google.datalab.bigquery.Query.resolve_parameters(
      {'key%(custom_key)s': 'value%(custom_key)s'}, params),
      {'keycustom_value': 'valuecustom_value'})
    params = [{'name': '_ds', 'value': 'custom_value'}]
    self.assertDictEqual(google.datalab.bigquery.Query.resolve_parameters(
      {'key%(_ds)s': 'value%(_ds)s'}, params),
      {'keycustom_value': 'valuecustom_value'})
