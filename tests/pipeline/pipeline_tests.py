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

import datetime
import mock
import re
import unittest
import yaml
from pytz import timezone

import gcloud.storage
import google.auth
import google.datalab
import google.datalab.bigquery as bq
import google.datalab.contrib.pipeline._pipeline as pipeline

import google.auth


class PipelineTest(unittest.TestCase):

  _test_pipeline_yaml_spec = """
schedule:
  start: 2009-05-05T22:28:15Z
  end: 2009-05-06T22:28:15Z
  interval: '0-59 * * * *'
tasks:
  current_timestamp:
    type: bq.execute
    query: $foo_query
    use_legacy_sql: False
  tomorrows_timestamp:
    type: bq.execute
    query: $foo_query
    use_legacy_sql: False
    up_stream:
      - current_timestamp
"""

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = mock.Mock(spec=google.auth.credentials.Credentials)
    return google.datalab.Context(project_id, creds)

  def test_get_dependency_definition_single(self):
    dependencies = pipeline.Pipeline._get_dependency_definition('t2', ['t1'])
    self.assertEqual(dependencies, 't2.set_upstream(t1)\n')

  def test_get_dependency_definition_multiple(self):
    dependencies = pipeline.Pipeline._get_dependency_definition('t2', ['t1', 't3'])
    self.assertEqual(dependencies, 't2.set_upstream(t1)\nt2.set_upstream(t3)\n')

  def test_get_bash_operator_definition(self):
    task_id = 'print_pdt_date'
    task_details = {}
    task_details['type'] = 'Bash'
    task_details['bash_command'] = 'date'
    operator_def = pipeline.Pipeline(None, None)._get_operator_definition(task_id, task_details)
    self.assertEqual(
        operator_def,
        'print_pdt_date = BashOperator(task_id=\'print_pdt_date_id\', '
        'bash_command=\'date\', dag=dag)\n')

  def test_get_templated_bash_operator_definition(self):
    task_id = 'foo_task'
    task_details = {}
    task_details['type'] = 'Bash'
    task_details['bash_command'] = 'echo "{{ ds }}"'
    operator_def = pipeline.Pipeline(None, None)._get_operator_definition(task_id, task_details)
    self.assertEqual(
      operator_def,
      """foo_task = BashOperator(task_id='foo_task_id', bash_command='echo "{{ ds }}"', dag=dag)
""")  # noqa

  def test_get_templated_bash_bq_definition(self):
    task_id = 'foo_task'
    task_details = {}
    task_details['type'] = 'BigQuery'
    task_details['query'] = google.datalab.bigquery.Query(
      'SELECT * FROM `cloud-datalab-samples.httplogs.logs_{{ ds }}`')
    operator_def = pipeline.Pipeline(None, None)._get_operator_definition(task_id, task_details)
    self.assertEqual(
      operator_def,
      """foo_task = BigQueryOperator(task_id='foo_task_id', bql='SELECT * FROM `cloud-datalab-samples.httplogs.logs_{{ ds }}`', use_legacy_sql=False, dag=dag)
""")  # noqa

  @mock.patch('google.datalab.bigquery.commands._bigquery._get_table')
  def test_get_bq_execute_operator_definition(self, mock_table):
    mock_table.return_value = bq.Table(
        'foo_project.foo_dataset.foo_table',
        context=PipelineTest._create_context())
    task_id = 'foo'
    task_details = {}
    task_details['type'] = 'BigQuery'
    task_details['query'] = google.datalab.bigquery.Query(
      'SELECT * FROM publicdata.samples.wikipedia LIMIT 5')
    operator_def = pipeline.Pipeline(None, None)._get_operator_definition(
        task_id, task_details)
    self.assertEqual(operator_def, "foo = BigQueryOperator(task_id='foo_id', bql='SELECT * FROM publicdata.samples.wikipedia LIMIT 5', use_legacy_sql=False, dag=dag)\n")  # noqa

  @mock.patch('google.datalab.bigquery.commands._bigquery._get_table')
  def test_get_bq_extract_operator_definition(self, mock_table):
    mock_table.return_value = bq.Table(
        'foo_project.foo_dataset.foo_table',
        context=PipelineTest._create_context())
    task_id = 'foo'
    task_details = {}
    task_details['type'] = 'BigQueryToCloudStorage'
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
    mock_table.return_value = bq.Table(
        'foo_project.foo_dataset.foo_table',
        context=PipelineTest._create_context())
    task_id = 'foo'
    task_details = {}
    task_details['type'] = 'GoogleCloudStorageToBigQuery'
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

  def test_get_pydatalab_bq_load_operator_definition(self):
    task_id = 'bq_pipeline_load_task'
    task_details = {}
    task_details['type'] = 'pydatalab.bq.load'
    task_details['delimiter'] = ','
    task_details['format'] = 'csv'
    task_details['mode'] = 'create'
    task_details['path'] = 'test/path'
    task_details['quote'] = '"'
    schema = [
      {
        'mode': 'NULLABLE',
        'type': 'int64',
        'description': 'description1',
        'name': 'col1',
      },
      {
        'mode': 'required',
        'type': 'STRING',
        'description': 'description1',
        'name': 'col2',
      }
    ]
    task_details['schema'] = schema
    task_details['skip'] = 0
    task_details['strict'] = True
    task_details['table'] = 'project.test.table'

    actual = pipeline.Pipeline(None, None)._get_operator_definition(
        task_id, task_details)
    pattern = re.compile("""bq_pipeline_load_task = LoadOperator\(task_id='bq_pipeline_load_task_id', delimiter=',', format='csv', mode='create', path='test/path', quote='"', schema=(.*), skip=0, strict=True, table='project.test.table', dag=dag\)""")  # noqa

    # group(1) has the string that follows the "schema=", i.e. the list of dicts.
    self.assertEqual(pattern.match(actual).group(1), str(schema))

  def test_get_pydatalab_bq_execute_operator_definition(self):
    task_id = 'bq_pipeline_execute_task'
    task_details = {}
    task_details['type'] = 'pydatalab.bq.execute'
    task_details['large'] = True
    task_details['mode'] = 'create'
    task_details['query'] = 'foo_query'
    task_details['table'] = 'project.test.table'

    actual = pipeline.Pipeline(None, None)._get_operator_definition(task_id, task_details)
    expected = """bq_pipeline_execute_task = ExecuteOperator(task_id='bq_pipeline_execute_task_id', large=True, mode='create', query='foo_query', table='project.test.table', dag=dag)
"""  # noqa
    self.assertEqual(actual, expected)

  def test_get_pydatalab_bq_extract_operator_definition(self):
    task_id = 'bq_pipeline_extract_task'
    task_details = {}
    task_details['type'] = 'pydatalab.bq.extract'
    task_details['billing'] = 'foo'
    task_details['compress'] = True
    task_details['delimiter'] = ','
    task_details['format'] = 'csv'
    task_details['header'] = True
    task_details['path'] = 'test/path'

    actual = pipeline.Pipeline(None, None)._get_operator_definition(task_id, task_details)
    expected = """bq_pipeline_extract_task = ExtractOperator(task_id='bq_pipeline_extract_task_id', billing='foo', compress=True, delimiter=',', format='csv', header=True, path='test/path', dag=dag)
"""  # noqa
    self.assertEqual(actual, expected)

  def test_get_query_params(self):
    task_details = {}
    task_details['parameters'] = [
        {
          'name': 'endpoint',
          'type': 'STRING',
          'value': 'Interact3'
        },
        {
          'name': 'table_name',
          'type': 'STRING',
          'value': 'cloud-datalab-samples.httplogs.logs_20140615'
        }
    ]
    actual = pipeline.Pipeline._get_query_parameters(task_details['parameters'])
    expected = [
      {
        'name': 'endpoint',
        'parameterType': {
          'type': 'STRING'
        },
        'parameterValue': {
          'value': 'Interact3'
        }
      },
      {
        'name': 'table_name',
        'parameterType': {
         'type': 'STRING'
        },
        'parameterValue': {
          'value': 'cloud-datalab-samples.httplogs.logs_20140615'
        }
      }
    ]
    self.assertListEqual(actual, expected)

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

  def test_get_random_operator_classname(self):
    self.assertEqual(pipeline.Pipeline._get_operator_classname('Unknown'),
                     'UnknownOperator')

  def test_get_dag_definition(self):
    test_pipeline = pipeline.Pipeline('foo', None)
    self.assertEqual(test_pipeline._get_dag_definition('bar'),
                     'dag = DAG(dag_id=\'foo\', schedule_interval=\'bar\', '
                     'default_args=default_args)\n\n')

  def test_get_datetime_expr(self):
    dag_dict = yaml.load(PipelineTest._test_pipeline_yaml_spec)
    start = dag_dict.get('schedule').get('start')
    datetime_expr = pipeline.Pipeline._get_datetime_expr_str(start)

    self.assertEqual(datetime_expr,
                     'datetime.datetime.strptime(\'2009-05-05T22:28:15\', '
                     '\'%Y-%m-%dT%H:%M:%S\').replace(tzinfo=timezone(\'UTC\'))')
    self.assertEqual(eval(datetime_expr),
                     datetime.datetime(2009, 5, 5, 22, 28, 15,
                                       tzinfo=timezone('UTC')))

  @mock.patch('gcloud.storage.Blob')
  def test_write_to_gcs(self, mock_blob_class):
    mock_blob = mock_blob_class.return_value
    dag_dict = yaml.load(PipelineTest._test_pipeline_yaml_spec)
    test_pipeline = pipeline.Pipeline('foo_pipeline', dag_dict)
    test_pipeline.write_to_gcs()
    mock_blob.upload_from_string.assert_called_with(test_pipeline._get_airflow_spec())

  def test_get_default_args(self):
    dag_dict = yaml.load(PipelineTest._test_pipeline_yaml_spec)
    self.assertEqual(
      pipeline.Pipeline._get_default_args(dag_dict.get('schedule').get('start'),
                                          dag_dict.get('schedule').get('end')),
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


if __name__ == '__main__':
  unittest.main()
