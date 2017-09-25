#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc. All rights reserved.
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

import google
import google.auth
import mock
import re
import unittest


class TestCases(unittest.TestCase):

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = mock.Mock(spec=google.auth.credentials.Credentials)
    return google.datalab.Context(project_id, creds)

  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.utils.commands.notebook_environment')
  @mock.patch('google.datalab.bigquery.Table.exists')
  @mock.patch('google.datalab.bigquery.commands._bigquery._get_table')
  def test_pipeline_cell(self, mock_get_table, mock_table_exists, mock_environment,
                         mock_default_context):
    table = google.datalab.bigquery.Table('project.test.table')
    mock_get_table.return_value = table
    mock_table_exists.return_value = True
    context = TestCases._create_context()
    mock_default_context.return_value = context

    env = {
      'foo_query': google.datalab.bigquery.Query(
        'SELECT * FROM publicdata.samples.wikipedia LIMIT 5')
    }
    mock_environment.return_value = env

    args = {'name': 'bq_pipeline_test', 'debug': True, 'billing': 'foo_billing'}
    # TODO(rajivpb): The references to foo_query need to be resolved.
    cell_body = """
            email: foo@bar.com
            schedule:
                start_date: 2009-05-05T22:28:15Z
                end_date: 2009-05-06T22:28:15Z
                schedule_interval: '@hourly'
            input:
                path: test/path
                table: project.test.table
                schema:
                    - name: col1
                      type: int64
                      mode: NULLABLE
                      description: description1
                    - name: col2
                      type: STRING
                      mode: required
                      description: description1
            transformation:
                query: foo_query
            output:
                path: test/path
                table: project.test.table
       """

    output = google.datalab.contrib.bigquery.commands._bigquery._pipeline_cell(args, cell_body)

    pattern = re.compile("""
import datetime
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.contrib.operators.bigquery_operator import BigQueryOperator
from airflow.contrib.operators.bigquery_table_delete_operator import BigQueryTableDeleteOperator
from airflow.contrib.operators.bigquery_to_bigquery import BigQueryToBigQueryOperator
from airflow.contrib.operators.bigquery_to_gcs import BigQueryToCloudStorageOperator
from airflow.contrib.operators.gcs_to_bq import GoogleCloudStorageToBigQueryOperator
from google.datalab.contrib.bigquery.operators.bq_load_operator import LoadOperator
from google.datalab.contrib.bigquery.operators.bq_execute_operator import ExecuteOperator
from google.datalab.contrib.bigquery.operators.bq_extract_operator import ExtractOperator
from datetime import timedelta
from pytz import timezone

default_args = {
    'owner': 'Datalab',
    'depends_on_past': False,
    'email': \['foo@bar.com'\],
    'start_date': datetime.datetime.strptime\('2009-05-05T22:28:15', '%Y-%m-%dT%H:%M:%S'\).replace\(tzinfo=timezone\('UTC'\)\),
    'end_date': datetime.datetime.strptime\('2009-05-06T22:28:15', '%Y-%m-%dT%H:%M:%S'\).replace\(tzinfo=timezone\('UTC'\)\),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta\(minutes=1\),
}

dag = DAG\(dag_id='bq_pipeline_test', schedule_interval='@hourly', default_args=default_args\)

bq_pipeline_execute_task = ExecuteOperator\(task_id='bq_pipeline_execute_task_id', cell_args=(.*), mode='create', query='foo_query', table='project.test.table', dag=dag\)
bq_pipeline_extract_task = ExtractOperator\(task_id='bq_pipeline_extract_task_id', compress=True, delimiter=',', format='csv', header=True, path='test/path', dag=dag\)
bq_pipeline_load_task = LoadOperator\(task_id='bq_pipeline_load_task_id', delimiter=',', format='csv', mode='create', path='test/path', quote='"', schema=(.*), skip=0, strict=True, table='project.test.table', dag=dag\)
bq_pipeline_execute_task.set_upstream\(bq_pipeline_load_task\)
bq_pipeline_extract_task.set_upstream\(bq_pipeline_execute_task\)
""")  # noqa

    self.assertIsNotNone(pattern.match(output))

    # group(1) has the string that follows the "cell_args=", i.e. the list of dicts.
    actual_args_str = pattern.match(output).group(1)
    expected_args_str = '{0}'.format(args)
    self.assertEqual(expected_args_str, actual_args_str)

    # group(2) has the string that follows the "schema=", i.e. the list of dicts.
    actual_schema_str = pattern.match(output).group(2)
    self.assertIn("'type': 'int64'", actual_schema_str)
    self.assertIn("'mode': 'NULLABLE'", actual_schema_str)
    self.assertIn("'name': 'col1'", actual_schema_str)
    self.assertIn("'description': 'description1'", actual_schema_str)
    self.assertIn("'type': 'STRING'", actual_schema_str)
    self.assertIn("'mode': 'required'", actual_schema_str)
    self.assertIn("'name': 'col2'", actual_schema_str)
    self.assertIn("'description': 'description1'", actual_schema_str)