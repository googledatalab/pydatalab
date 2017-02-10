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
import mock
from oauth2client.client import AccessTokenCredentials
import unittest

import google.datalab
import google.datalab.bigquery


class TestCases(unittest.TestCase):

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
    results = q.execute().result()

    self.assertEqual(sql, results.sql)
    self.assertEqual('(%s)' % sql, q._repr_sql_())
    self.assertEqual(sql, str(q))
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
    results = q.execute().result()

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
    results = q.execute().result()

    self.assertEqual(1, results.length)
    self.assertEqual('test_job', results.job_id)

  @mock.patch('google.datalab.bigquery._api.Api.jobs_insert_query')
  def test_malformed_response_raises_exception(self, mock_api_insert_query):
    mock_api_insert_query.return_value = {}

    q = TestCases._create_query()

    with self.assertRaises(Exception) as error:
      _ = q.execute().result()
    self.assertEqual('Unexpected response from server', str(error.exception))

  def test_nested_subquery_expansion(self):
    # test expanding subquery and udf validation
    with self.assertRaises(Exception) as error:
      TestCases._create_query('SELECT * FROM subquery', subqueries=['subquery'])

    with self.assertRaises(Exception) as error:
      TestCases._create_query('SELECT test_udf(field1) FROM test_table', udfs=['test_udf'])

    values = {}

    # test direct subquery expansion
    q1 = TestCases._create_query('SELECT * FROM test_table', name='q1', values=values)
    q2 = TestCases._create_query('SELECT * FROM q1', name='q2', subqueries=['q1'], values=values)
    self.assertEqual('WITH q1 AS (SELECT * FROM test_table)\nSELECT * FROM q1', q2.sql)

    # test recursive, second level subquery expansion
    q3 = TestCases._create_query('SELECT * FROM q2', name='q3', subqueries=['q2'], values=values)
    # subquery listing order is random, try both possibilities
    expected_sql1 = 'WITH q1 AS (%s),\nq2 AS (%s)\n%s' % (q1._sql, q2._sql, q3._sql)
    expected_sql2 = 'WITH q2 AS (%s),\nq1 AS (%s)\n%s' % (q2._sql, q1._sql, q3._sql)

    self.assertTrue((expected_sql1 == q3.sql) or (expected_sql2 == q3.sql))

  @staticmethod
  def _create_query(sql='SELECT * ...', name=None, values=None, udfs=None, data_sources=None,
                    subqueries=None):
    q = google.datalab.bigquery.Query(sql, values=values, udfs=udfs, data_sources=data_sources,
                                      subqueries=subqueries)
    if name:
      values[name] = q
    return q

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = AccessTokenCredentials('test_token', 'test_ua')
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
