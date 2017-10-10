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
import google.datalab.contrib.bigquery.commands._bigquery as bq
import mock
import re
import unittest


class TestCases(unittest.TestCase):

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = mock.Mock(spec=google.auth.credentials.Credentials)
    return google.datalab.Context(project_id, creds)

  def test_get_load_parameters(self):
    input_config = {
      'path': 'test_path',
      'table': 'test_table',
      'schema': 'test_schema',
      'csv': {
        'delimiter': ';',
        'skip': 9
      },
    }
    actual_load_config = bq._get_load_parameters(input_config)
    expected_load_config = {
      'type': 'pydatalab.bq.load',
      'format': 'csv',
      'path': 'test_path',
      'table': 'test_table',
      'schema': 'test_schema',
      'mode': 'create',
      'csv_options': {
        'delimiter': ';',
        'skip': 9
      }
    }
    self.assertDictEqual(actual_load_config, expected_load_config)

    input_config = {
      'path': 'test_path',
      'table': 'test_table',
    }
    actual_load_config = bq._get_load_parameters(input_config)
    self.assertEqual(actual_load_config['mode'], 'append')

    input_config = {
      'table': 'test_table',
      'schema': 'test_schema'
    }
    actual_load_config = bq._get_load_parameters(input_config)
    self.assertIsNone(actual_load_config)

    input_config = {
      'schema': 'test_schema'
    }
    actual_load_config = bq._get_load_parameters(input_config)
    self.assertIsNone(actual_load_config)

    input_config = {
      'path': 'test_path',
    }
    actual_load_config = bq._get_load_parameters(input_config)
    expected_load_config = {
      'type': 'pydatalab.bq.load',
      'format': 'csv',
      'csv_options': None,
      'path': 'test_path',
    }
    self.assertDictEqual(actual_load_config, expected_load_config)

    input_config = {
      'path': 'test_path',
      'format': 'json'
    }
    actual_load_config = bq._get_load_parameters(input_config)
    expected_load_config = {
      'type': 'pydatalab.bq.load',
      'format': 'json',
      'path': 'test_path',
      'csv_options': None
    }
    self.assertDictEqual(actual_load_config, expected_load_config)

  def test_get_extract_parameters(self):
    input_config = {
      'path': 'test_path',
      'table': 'test_table',
    }
    execute_config = {}
    actual_extract_config = bq._get_extract_parameters('foo_execute_task', execute_config,
                                                       input_config)
    expected_extract_config = {
      'type': 'pydatalab.bq.extract',
      'up_stream': ['foo_execute_task'],
      'format': 'csv',
      'csv_options': None,
      'path': 'test_path',
      'table': 'test_table',
    }

    self.assertDictEqual(actual_extract_config, expected_extract_config)
    self.assertDictEqual(execute_config, {'table': 'test_table'})

  @mock.patch('google.cloud.storage.Client')
  @mock.patch('google.datalab.Context.default')
  @mock.patch('google.datalab.utils.commands.notebook_environment')
  @mock.patch('google.datalab.bigquery.Table.exists')
  @mock.patch('google.datalab.bigquery.commands._bigquery._get_table')
  @mock.patch('google.cloud.storage.Blob')
  @mock.patch('google.cloud.storage.Client.get_bucket')
  def test_pipeline_cell_golden(self, mock_client_get_bucket, mock_blob_class, mock_get_table,
                                mock_table_exists, mock_environment, mock_default_context,
                                mock_client):
    table = google.datalab.bigquery.Table('project.test.table')
    mock_get_table.return_value = table
    mock_table_exists.return_value = True
    context = TestCases._create_context()
    mock_default_context.return_value = context
    mock_client_get_bucket.return_value = mock.Mock(spec=google.cloud.storage.Bucket)
    mock_blob = mock_blob_class.return_value

    env = {
      'foo_query': google.datalab.bigquery.Query(
        'SELECT * FROM publicdata.samples.wikipedia LIMIT 5')
    }
    mock_environment.return_value = env

    args = {'name': 'bq_pipeline_test', 'debug': True}
    # TODO(rajivpb): The references to foo_query need to be resolved.
    cell_body = """
            emails: foo1@test.com,foo2@test.com
            schedule:
                start: 2009-05-05T22:28:15Z
                end: 2009-05-06T22:28:15Z
                interval: '@hourly'
            input:
                path: test/path
                table: project.test.table
                csv:
                  header: True
                  strict: False
                  quote: '"'
                  skip: 5
                  delimiter: ','
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

    output = bq._pipeline_cell(args, cell_body)

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
    'email': \['foo1@test.com', 'foo2@test.com'\],
    'start_date': datetime.datetime.strptime\('2009-05-05T22:28:15', '%Y-%m-%dT%H:%M:%S'\).replace\(tzinfo=timezone\('UTC'\)\),
    'end_date': datetime.datetime.strptime\('2009-05-06T22:28:15', '%Y-%m-%dT%H:%M:%S'\).replace\(tzinfo=timezone\('UTC'\)\),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta\(minutes=1\),
}

dag = DAG\(dag_id='bq_pipeline_test', schedule_interval='@hourly', default_args=default_args\)

bq_pipeline_execute_task = ExecuteOperator\(task_id='bq_pipeline_execute_task_id', mode='create', query='foo_query', table='project.test.table', dag=dag\)
bq_pipeline_extract_task = ExtractOperator\(task_id='bq_pipeline_extract_task_id', csv_options=None, format='csv', path='test/path', table='project.test.table', dag=dag\)
bq_pipeline_load_task = LoadOperator\(task_id='bq_pipeline_load_task_id', csv_options=(.*), format='csv', mode='create', path='test/path', schema=(.*), table='project.test.table', dag=dag\)
bq_pipeline_execute_task.set_upstream\(bq_pipeline_load_task\)
bq_pipeline_extract_task.set_upstream\(bq_pipeline_execute_task\)
""")  # noqa

    self.assertIsNotNone(pattern.match(output))

    # group(1) has the string that follows the "csv_options=", for the load operator.
    actual_csv_options_dict_str = pattern.match(output).group(1)
    self.assertIn("\'header\': True", actual_csv_options_dict_str)
    self.assertIn("\'delimiter\': \',\'", actual_csv_options_dict_str)
    self.assertIn("\'skip\': 5", actual_csv_options_dict_str)
    self.assertIn("\'strict\': False", actual_csv_options_dict_str)
    self.assertIn("\'quote\': \'\"\'", actual_csv_options_dict_str)

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

    mock_blob.upload_from_string.assert_called_with(output)
