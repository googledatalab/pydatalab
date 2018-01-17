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

import google.auth
import google.datalab
import google.datalab.bigquery as bq
import google.datalab.contrib.pipeline._pipeline as pipeline

import google.auth


class PipelineTest(unittest.TestCase):

  _test_pipeline_yaml_spec = """
emails: foo1@test.com,foo2@test.com
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
    project_id = 'project'
    creds = mock.Mock(spec=google.auth.credentials.Credentials)
    return google.datalab.Context(project_id, creds)

  def test_get_dependency_definition_single(self):
    dependencies = pipeline.PipelineGenerator._get_dependency_definition('t2', ['t1'])
    self.assertEqual(dependencies, 't2.set_upstream(t1)\n')

  def test_get_dependency_definition_multiple(self):
    dependencies = pipeline.PipelineGenerator._get_dependency_definition('t2', ['t1', 't3'])
    self.assertEqual(dependencies, 't2.set_upstream(t1)\nt2.set_upstream(t3)\n')

  def test_get_bash_operator_definition(self):
    task_id = 'print_pdt_date'
    task_details = {}
    task_details['type'] = 'Bash'
    task_details['bash_command'] = 'date'

    operator_def = pipeline.PipelineGenerator._get_operator_definition(task_id, task_details, None)
    self.assertEqual(operator_def, """print_pdt_date = BashOperator(task_id=\'print_pdt_date_id\', bash_command=\"\"\"date\"\"\", dag=dag)
""")  # noqa

  def test_get_bash_operator_definition_with_templates(self):
    task_id = 'print_pdt_date'
    task_details = {}
    task_details['type'] = 'Bash'
    task_details['output_encoding'] = 'utf-8'
    task_details['bash_command'] = 'date_%(_ds)s'
    operator_def = pipeline.PipelineGenerator._get_operator_definition(task_id, task_details, None)
    self.assertEqual(operator_def, """print_pdt_date = BashOperator(task_id=\'print_pdt_date_id\', bash_command=\"\"\"date_{{ ds }}\"\"\", output_encoding=\"\"\"utf-8\"\"\", dag=dag)
""")  # noqa

    # Airflow macros should get replaced in templated fields
    task_details['bash_command'] = 'date_%(_ds)s'
    operator_def = pipeline.PipelineGenerator._get_operator_definition(task_id, task_details, None)
    self.assertEqual(operator_def, """print_pdt_date = BashOperator(task_id=\'print_pdt_date_id\', bash_command=\"\"\"date_{{ ds }}\"\"\", output_encoding=\"\"\"utf-8\"\"\", dag=dag)
""")  # noqa

    # Airflow macros should not get replaced in non-templated fields
    task_details['bash_command'] = 'date'
    task_details['output_encoding'] = 'foo_%(_ds)s'
    operator_def = pipeline.PipelineGenerator._get_operator_definition(task_id, task_details, None)
    self.assertEqual(operator_def, """print_pdt_date = BashOperator(task_id=\'print_pdt_date_id\', bash_command=\"\"\"date\"\"\", output_encoding=\"\"\"foo_%(_ds)s\"\"\", dag=dag)
""")  # noqa

    # User-defined modifiers should get replaced in templated fields
    task_details['bash_command'] = 'date_%(foo_key)s'
    operator_def = pipeline.PipelineGenerator._get_operator_definition(
      task_id, task_details, [{'name': 'foo_key', 'value': 'foo_value', 'type': 'STRING'}])
    self.assertEqual(operator_def, """print_pdt_date = BashOperator(task_id=\'print_pdt_date_id\', bash_command=\"\"\"date_foo_value\"\"\", output_encoding=\"\"\"foo_%(_ds)s\"\"\", dag=dag)
""")  # noqa

    # User-defined modifiers should take precedence over the built-in airflow macros
    task_details['bash_command'] = 'date_%(_ds)s'
    operator_def = pipeline.PipelineGenerator._get_operator_definition(
      task_id, task_details, [{'name': '_ds', 'value': 'user_value', 'type': 'STRING'}])
    self.assertEqual(operator_def, """print_pdt_date = BashOperator(task_id=\'print_pdt_date_id\', bash_command=\"\"\"date_user_value\"\"\", output_encoding=\"\"\"foo_%(_ds)s\"\"\", dag=dag)
""")  # noqa

  @mock.patch('google.datalab.bigquery.commands._bigquery._get_table')
  def test_get_bq_execute_operator_definition(self, mock_table):
    mock_table.return_value = bq.Table(
        'foo_project.foo_dataset.foo_table',
        context=PipelineTest._create_context())
    task_id = 'foo'
    task_details = {}
    task_details['type'] = 'BigQuery'

    # Adding newlines to the query to mimic actual usage of %%bq query ...
    task_details['query'] = google.datalab.bigquery.Query("""SELECT *
FROM publicdata.samples.wikipedia
LIMIT 5""")
    operator_def = pipeline.PipelineGenerator._get_operator_definition(task_id, task_details, None)
    self.assertEqual(operator_def, """foo = BigQueryOperator(task_id='foo_id', bql=\"\"\"SELECT *\nFROM publicdata.samples.wikipedia\nLIMIT 5\"\"\", use_legacy_sql=False, dag=dag)
""")  # noqa

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
    operator_def = pipeline.PipelineGenerator._get_operator_definition(task_id, task_details, None)
    self.assertEqual(operator_def, """foo = BigQueryToCloudStorageOperator(task_id='foo_id', compression=\"\"\"GZIP\"\"\", destination_cloud_storage_uris=[\'foo_path\'], export_format=\"\"\"CSV\"\"\", field_delimiter=\"\"\"$\"\"\", print_header=False, source_project_dataset_table=\"\"\"foo_project.foo_dataset.foo_table\"\"\", dag=dag)
""")  # noqa

    task_details['format'] = 'json'
    operator_def = pipeline.PipelineGenerator._get_operator_definition(task_id, task_details, None)
    self.assertEqual(operator_def, """foo = BigQueryToCloudStorageOperator(task_id='foo_id', compression=\"\"\"GZIP\"\"\", destination_cloud_storage_uris=[\'foo_path\'], export_format=\"\"\"NEWLINE_DELIMITED_JSON\"\"\", field_delimiter=\"\"\"$\"\"\", print_header=False, source_project_dataset_table=\"\"\"foo_project.foo_dataset.foo_table\"\"\", dag=dag)
""")  # noqa

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
    operator_def = pipeline.PipelineGenerator._get_operator_definition(task_id, task_details, None)
    self.assertEqual(operator_def, """foo = GoogleCloudStorageToBigQueryOperator(task_id='foo_id', bucket=\"\"\"foo_bucket\"\"\", destination_project_dataset_table=\"\"\"foo_project.foo_dataset.foo_table\"\"\", export_format=\"\"\"CSV\"\"\", field_delimiter=\"\"\"$\"\"\", skip_leading_rows=False, source_objects=\"\"\"foo_file.csv\"\"\", dag=dag)
""")  # noqa

    task_details['format'] = 'json'
    operator_def = pipeline.PipelineGenerator._get_operator_definition(task_id, task_details, None)
    self.assertEqual(operator_def, """foo = GoogleCloudStorageToBigQueryOperator(task_id='foo_id', bucket=\"\"\"foo_bucket\"\"\", destination_project_dataset_table=\"\"\"foo_project.foo_dataset.foo_table\"\"\", export_format=\"\"\"NEWLINE_DELIMITED_JSON\"\"\", field_delimiter=\"\"\"$\"\"\", skip_leading_rows=False, source_objects=\"\"\"foo_file.csv\"\"\", dag=dag)
""")  # noqa

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

    actual = pipeline.PipelineGenerator._get_operator_definition(task_id, task_details, None)
    pattern = re.compile("""bq_pipeline_load_task = LoadOperator\(task_id='bq_pipeline_load_task_id', delimiter=\"\"\",\"\"\", format=\"\"\"csv\"\"\", mode=\"\"\"create\"\"\", path=\"\"\"test/path\"\"\", quote=\"\"\""\"\"\", schema=(.*), skip=0, strict=True, table=\"\"\"project.test.table\"\"\", dag=dag\)""")  # noqa
    # group(1) has the string that follows the "schema=", i.e. the list of dicts.
    # Since we're comparing string literals of dicts that have the items re-ordered, we just sort
    # the string. This is a loose check.
    sorted_string_of_actual_schema = ''.join(sorted(pattern.match(actual).group(1)))
    sorted_string_of_expected_schema = ''.join(sorted(str(schema)))
    self.assertEqual(sorted_string_of_actual_schema, sorted_string_of_expected_schema)

  def test_get_pydatalab_bq_execute_operator_definition(self):
    task_id = 'bq_pipeline_execute_task'
    task_details = {}
    task_details['type'] = 'pydatalab.bq.execute'
    task_details['large'] = True
    task_details['mode'] = 'create'
    task_details['sql'] = 'foo_query'
    task_details['table'] = 'project.test.table'
    actual = pipeline.PipelineGenerator._get_operator_definition(task_id, task_details, None)
    expected = """bq_pipeline_execute_task = ExecuteOperator(task_id='bq_pipeline_execute_task_id', large=True, mode=\"\"\"create\"\"\", sql=\"\"\"foo_query\"\"\", table=\"\"\"project.test.table\"\"\", dag=dag)
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

    actual = pipeline.PipelineGenerator._get_operator_definition(task_id, task_details, None)
    expected = """bq_pipeline_extract_task = ExtractOperator(task_id='bq_pipeline_extract_task_id', billing=\"\"\"foo\"\"\", compress=True, delimiter=\"\"\",\"\"\", format=\"\"\"csv\"\"\", header=True, path=\"\"\"test/path\"\"\", dag=dag)
"""  # noqa
    self.assertEqual(actual, expected)

  def test_get_unknown_operator_definition(self):
    task_id = 'id'
    task_details = {}
    task_details['type'] = 'Unknown'
    task_details['foo'] = 'bar'
    task_details['bar_typed'] = False
    operator_def = pipeline.PipelineGenerator._get_operator_definition(task_id, task_details, None)
    self.assertEqual(operator_def,
                     'id = UnknownOperator(''task_id=\'id_id\', ' +
                     'bar_typed=False, foo="""bar""", dag=dag)\n')

  def test_get_random_operator_class_name(self):
    self.assertEqual(pipeline.PipelineGenerator._get_operator_class_name('Unknown'),
                     ('UnknownOperator', 'google.datalab.contrib.pipeline._pipeline'))

  def test_get_dag_definition(self):
    self.assertEqual(pipeline.PipelineGenerator._get_dag_definition('foo', 'bar', ),
                     'dag = DAG(dag_id=\'foo\', schedule_interval=\'bar\', '
                     'catchup=False, default_args=default_args)\n\n')

    self.assertEqual(pipeline.PipelineGenerator._get_dag_definition('foo', 'bar', True),
                     'dag = DAG(dag_id=\'foo\', schedule_interval=\'bar\', '
                     'catchup=True, default_args=default_args)\n\n')

  def test_get_datetime_expr(self):
    dag_dict = yaml.load(PipelineTest._test_pipeline_yaml_spec)
    start = dag_dict.get('schedule').get('start')
    datetime_expr = pipeline.PipelineGenerator._get_datetime_expr_str(start)

    self.assertEqual(datetime_expr,
                     'datetime.datetime.strptime(\'2009-05-05T22:28:15\', \'%Y-%m-%dT%H:%M:%S\')')
    self.assertEqual(eval(datetime_expr), datetime.datetime(2009, 5, 5, 22, 28, 15))

  def test_get_default_args(self):
    actual = pipeline.PipelineGenerator._get_default_args({}, None)
    self.assertIn("'end_date': None", actual)
    self.assertIn("'start_date': datetime.datetime.strptime(", actual)
    self.assertIn("'email': []", actual)
    self.assertIn("'owner': 'Google Cloud Datalab'", actual)

    dag_dict = yaml.load(PipelineTest._test_pipeline_yaml_spec)
    dag_dict['schedule']['retries'] = 5
    dag_dict['schedule']['email_on_retry'] = False
    dag_dict['schedule']['email_on_failure'] = False
    dag_dict['schedule']['retry_exponential_backoff'] = False
    dag_dict['schedule']['retry_delay_seconds'] = 10
    dag_dict['schedule']['max_retry_delay_seconds'] = 15
    actual = pipeline.PipelineGenerator._get_default_args(dag_dict.get('schedule'),
                                                          dag_dict.get('emails'))
    self.assertIn(
      "'end_date': datetime.datetime.strptime('2009-05-06T22:28:15', '%Y-%m-%dT%H:%M:%S')",
      actual)
    self.assertIn(
      "'start_date': datetime.datetime.strptime('2009-05-05T22:28:15', '%Y-%m-%dT%H:%M:%S')",
      actual)
    self.assertIn("'email': ['foo1@test.com', 'foo2@test.com']", actual)
    self.assertIn("'owner': 'Google Cloud Datalab'", actual)
    self.assertIn("'retries': 5", actual)
    self.assertIn("'email_on_retry': False", actual)
    self.assertIn("'email_on_failure': False", actual)
    self.assertIn("'retry_exponential_backoff': False", actual)
    self.assertIn("'retry_delay': timedelta(seconds=10)", actual)
    self.assertIn("'max_retry_delay': timedelta(seconds=15)", actual)

  def test_get_airflow_spec_with_default_schedule(self):
    dag_dict = yaml.load(PipelineTest._test_pipeline_yaml_spec)
    # We delete the schedule spec to test with defaults
    del dag_dict['schedule']

    actual = pipeline.PipelineGenerator.generate_airflow_spec('foo_name', dag_dict)
    self.assertIn('import datetime', actual)
    self.assertIn("'email': ['foo1@test.com', 'foo2@test.com']", actual)
    self.assertIn("schedule_interval='@once'", actual)
    self.assertIn('current_timestamp_id', actual)
    self.assertIn('tomorrows_timestamp_id', actual)
    self.assertIn('tomorrows_timestamp.set_upstream(current_timestamp)', actual)


if __name__ == '__main__':
  unittest.main()
