# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for google3.third_party.py.google.datalab.utils._file."""

from __future__ import absolute_import
from __future__ import unicode_literals

import datetime
import mock
import unittest
import yaml
from oauth2client.client import AccessTokenCredentials
from pytz import timezone

import google
import google.datalab.contrib.pipeline._pipeline as pipeline


class PipelineTest(unittest.TestCase):

  _dag_spec = """
email: foo@bar.com
schedule:
  start_date: 2009-05-05T22:28:15Z
  end_date: 2009-05-06T22:28:15Z
  schedule_interval: '0-59 * * * *'
tasks:
  current_timestamp:
    type: bq
    query: $foo_query
    use_legacy_sql: False
  tomorrows_timestamp:
    type: bq
    query: $foo_query
    use_legacy_sql: False
    up_stream:
      - current_timestamp
"""

  def test_get_dependency_definition_single(self):
    dependencies = pipeline.Pipeline._get_dependency_definition('t2', ['t1'])
    self.assertEqual(dependencies, 't2.set_upstream(t1)\n')

  def test_get_dependency_definition_multiple(self):
    dependencies = pipeline.Pipeline._get_dependency_definition('t2',
                                                                ['t1', 't3'])
    self.assertEqual(dependencies, 't2.set_upstream(t1)\nt2.set_upstream(t3)\n')

  def test_get_bash_operator_definition(self):
    task_id = 'print_pdt_date'
    task_details = {}
    task_details['type'] = 'bash'
    task_details['bash_command'] = 'date'
    operator_def = pipeline.Pipeline(None, None)._get_operator_definition(
        task_id, task_details)
    self.assertEqual(
        operator_def,
        'print_pdt_date = BashOperator(task_id=\'print_pdt_date_id\', '
        'bash_command=\'date\', dag=dag)\n')

  @mock.patch('google.datalab.bigquery.commands._bigquery._get_table')
  def test_get_bq_extract_operator_definition(self, mock_table):
    mock_table.return_value = google.datalab.bigquery.Table(
        'foo_project.foo_dataset.foo_table',
        context=PipelineTest._create_context())
    task_id = 'foo'
    task_details = {}
    task_details['type'] = 'bq.extract'
    task_details['table'] = 'foo_project.foo_dataset.foo_table'
    task_details['path'] = 'foo_path'
    task_details['format'] = 'csv'
    task_details['delimiter'] = '$'
    task_details['header'] = False
    task_details['compress'] = True
    operator_def = pipeline.Pipeline(None, None)._get_operator_definition(
        task_id, task_details)
    self.assertEqual(operator_def, ('foo = BigQueryToCloudStorageOperator(task_id=\'foo_id\', '
                                    'compression=\'GZIP\', destination_cloud_storage_uris=\''
                                    '[foo_path]\', export_format=\'CSV\', field_delimiter=\'$\', '
                                    'print_header=False, source_project_dataset_table=\''
                                    'foo_project.foo_dataset.foo_table\', dag=dag)\n'))

    task_details['format'] = 'json'
    operator_def = pipeline.Pipeline(None, None)._get_operator_definition(
        task_id, task_details)
    self.assertEqual(operator_def, ('foo = BigQueryToCloudStorageOperator(task_id=\'foo_id\', '
                                    'compression=\'GZIP\', destination_cloud_storage_uris=\''
                                    '[foo_path]\', export_format=\'NEWLINE_DELIMITED_JSON\', '
                                    'field_delimiter=\'$\', print_header=False, '
                                    'source_project_dataset_table=\''
                                    'foo_project.foo_dataset.foo_table\', dag=dag)\n'))

  @mock.patch('google.datalab.bigquery.commands._bigquery._get_table')
  def test_get_bq_load_operator_definition(self, mock_table):
    mock_table.return_value = google.datalab.bigquery.Table(
        'foo_project.foo_dataset.foo_table',
        context=PipelineTest._create_context())
    task_id = 'foo'
    task_details = {}
    task_details['type'] = 'bq.load'
    task_details['table'] = 'foo_project.foo_dataset.foo_table'
    task_details['path'] = 'gs://foo_bucket/foo_file.csv'
    task_details['format'] = 'csv'
    task_details['delimiter'] = '$'
    task_details['skip'] = False
    operator_def = pipeline.Pipeline(None, None)._get_operator_definition(
        task_id, task_details)
    self.assertEqual(operator_def, ('foo = GoogleCloudStorageToBigQueryOperator(task_id=\'foo_id\','
                                    ' bucket=\'foo_bucket\', destination_project_dataset_table='
                                    '\'foo_project.foo_dataset.foo_table\', export_format=\'CSV\', '
                                    'field_delimiter=\'$\', skip_leading_rows=False, '
                                    'source_objects=\'foo_file.csv\', dag=dag)\n'))

    task_details['format'] = 'json'
    operator_def = pipeline.Pipeline(None, None)._get_operator_definition(
        task_id, task_details)
    self.assertEqual(operator_def, ('foo = GoogleCloudStorageToBigQueryOperator(task_id=\'foo_id\','
                                    ' bucket=\'foo_bucket\', destination_project_dataset_table='
                                    '\'foo_project.foo_dataset.foo_table\', '
                                    'export_format=\'NEWLINE_DELIMITED_JSON\', '
                                    'field_delimiter=\'$\', skip_leading_rows=False, '
                                    'source_objects=\'foo_file.csv\', dag=dag)\n'))
  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = AccessTokenCredentials('test_token', 'test_ua')
    return google.datalab.Context(project_id, creds)

  def test_get_bq_operator_definition(self):
    task_id = 'query_wikipedia'
    task_details = {}
    task_details['type'] = 'bq'
    task_details['query'] = google.datalab.bigquery.Query(
        'SELECT * FROM publicdata.samples.wikipedia LIMIT 5')
    operator_def = pipeline.Pipeline(None, None)._get_operator_definition(
        task_id, task_details)
    self.assertEqual(
        operator_def,
        'query_wikipedia = BigQueryOperator(task_id=\'query_wikipedia_id\', '
        'bql=\'SELECT * FROM publicdata.samples.wikipedia LIMIT 5\', '
        'use_legacy_sql=False, dag=dag)\n')

  def test_get_unknown_operator_definition(self):
    task_id = 'id'
    task_details = {}
    task_details['type'] = 'Unknown'
    task_details['foo'] = 'bar'
    task_details['bar_typed'] = False
    operator_def = pipeline.Pipeline(None, None)._get_operator_definition(task_id, task_details)
    self.assertEqual(operator_def,
                     'id = UnknownOperator(''task_id=\'id_id\', ' +
                     'bar_typed=False, foo=\'bar\', dag=dag)\n')

  def test_get_operator_classname(self):
    self.assertEqual(pipeline.Pipeline._get_operator_classname('bash'), 'BashOperator')
    self.assertEqual(pipeline.Pipeline._get_operator_classname('bq'),
                     'BigQueryOperator')
    self.assertEqual(pipeline.Pipeline._get_operator_classname('Unknown'),
                     'UnknownOperator')

  def test_get_dag_definition(self):
    test_pipeline = pipeline.Pipeline('', 'foo')
    self.assertEqual(test_pipeline._get_dag_definition('bar'),
                     'dag = DAG(dag_id=\'foo\', schedule_interval=\'bar\', '
                     'default_args=default_args)\n\n')

  def test_get_datetime_expr(self):
    dag_dict = yaml.load(PipelineTest._dag_spec)
    start_date = dag_dict.get('schedule').get('start_date')
    datetime_expr = pipeline.Pipeline._get_datetime_expr_str(start_date)

    self.assertEqual(datetime_expr,
                     'datetime.datetime.strptime(\'2009-05-05T22:28:15\', '
                     '\'%Y-%m-%dT%H:%M:%S\').replace(tzinfo=timezone(\'UTC\'))')
    self.assertEqual(eval(datetime_expr),
                     datetime.datetime(2009, 5, 5, 22, 28, 15,
                                       tzinfo=timezone('UTC')))

  def test_get_default_args(self):
    import yaml
    dag_dict = yaml.load(PipelineTest._dag_spec)
    self.assertEqual(
        pipeline.Pipeline._get_default_args(
            dag_dict['email'],
            dag_dict.get('schedule').get('start_date'),
            dag_dict.get('schedule').get('end_date')),
"""
default_args = {
    'owner': 'Datalab',
    'depends_on_past': False,
    'email': ['foo@bar.com'],
    'start_date': datetime.datetime.strptime('2009-05-05T22:28:15', '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone('UTC')),
    'end_date': datetime.datetime.strptime('2009-05-06T22:28:15', '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone('UTC')),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

"""  # noqa
    )

  def test_py_bq(self):
    env = {}
    env['foo_query'] = google.datalab.bigquery.Query(
      'INSERT INTO rajivpb_demo.the_datetime_table (the_datetime) VALUES (CURRENT_DATETIME())')

    airflow_dag = pipeline.Pipeline(PipelineTest._dag_spec, 'demo_bq_dag_during_demo', env)
    expected_py = """
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.contrib.operators.bigquery_operator import BigQueryOperator
from airflow.contrib.operators.bigquery_table_delete_operator import BigQueryTableDeleteOperator
from airflow.contrib.operators.bigquery_to_bigquery import BigQueryToBigQueryOperator
from airflow.contrib.operators.bigquery_to_gcs import BigQueryToCloudStorageOperator
from airflow.contrib.operators.gcs_to_bq import GoogleCloudStorageToBigQueryOperator
from datetime import timedelta
from pytz import timezone

default_args = {
    'owner': 'Datalab',
    'depends_on_past': False,
    'email': ['foo@bar.com'],
    'start_date': datetime.datetime.strptime('2009-05-05T22:28:15', '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone('UTC')),
    'end_date': datetime.datetime.strptime('2009-05-06T22:28:15', '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone('UTC')),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(dag_id='demo_bq_dag_during_demo', schedule_interval='0-59 * * * *', default_args=default_args)

current_timestamp = BigQueryOperator(task_id='current_timestamp_id', bql='INSERT INTO rajivpb_demo.the_datetime_table (the_datetime) VALUES (CURRENT_DATETIME())', use_legacy_sql=False, dag=dag)
tomorrows_timestamp = BigQueryOperator(task_id='tomorrows_timestamp_id', bql='INSERT INTO rajivpb_demo.the_datetime_table (the_datetime) VALUES (CURRENT_DATETIME())', use_legacy_sql=False, dag=dag)
tomorrows_timestamp.set_upstream(current_timestamp)
""" # noqa
    self.assertEqual(airflow_dag.py, expected_py)


if __name__ == '__main__':
  unittest.main()
