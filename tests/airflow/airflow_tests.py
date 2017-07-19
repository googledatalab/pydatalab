# Copyright 2016 Google Inc. All Rights Reserved.
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
import unittest

from google.datalab.airflow import AirflowDag


class AirflowDagTest(unittest.TestCase):

  def test_get_dependency_definition_single(self):
    dependencies = AirflowDag._get_dependency_definition('t2', ['t1'])
    self.assertEqual(dependencies, 't2.set_upstream(t1)\n')

  def test_get_dependency_definition_multiple(self):
    dependencies = AirflowDag._get_dependency_definition('t2', ['t1', 't3'])
    self.assertEqual(dependencies, 't2.set_upstream(t1)\nt2.set_upstream(t3)\n')

  def test_get_bash_operator_definition(self):
    task_id = 'print_pdt_date'
    task_details = {}
    task_details['type'] = 'bash'
    task_details['bash_command'] = 'date'
    operator_def = AirflowDag._get_operator_definition(task_id, task_details)
    self.assertEqual(
        operator_def,
        'print_pdt_date = BashOperator(task_id=\'print_pdt_date_id\', bash_command=\'date\', dag=dag)\n')


  def test_get_bq_operator_definition(self):
    task_id = 'query_wikipedia'
    task_details = {}
    task_details['type'] = 'bq'
    task_details['query'] = 'SELECT * FROM publicdata.samples.wikipedia LIMIT 5'
    task_details['destination_dataset_table'] = False
    task_details['write_disposition'] = 'WRITE_EMPTY'
    task_details['allow_large_results'] = False
    task_details['bigquery_conn_id'] = 'bigquery_default'
    task_details['delegate_to'] = None
    task_details['udf_config'] = False
    task_details['use_legacy_sql'] = False
    operator_def = AirflowDag._get_operator_definition(task_id, task_details)
    self.assertEqual(
        operator_def,
        'query_wikipedia = BigQueryOperator(task_id=\'query_wikipedia_id\', delegate_to=None, udf_config=False, write_disposition=\'WRITE_EMPTY\', use_legacy_sql=False, destination_dataset_table=False, bql=\'SELECT * FROM publicdata.samples.wikipedia LIMIT 5\', bigquery_conn_id=\'bigquery_default\', allow_large_results=False, dag=dag)\n')

  def test_get_unknown_operator_definition(self):
    task_id = 'id'
    task_details = {}
    task_details['type'] = 'Unknown'
    task_details['foo'] = 'bar'
    task_details['bar_typed'] = False
    operator_def = AirflowDag._get_operator_definition(task_id, task_details)
    self.assertEqual(operator_def,
                     'id = UnknownOperator(''task_id=\'id_id\', ' +
                     'foo=\'bar\', bar_typed=False, dag=dag)\n')

  def test_get_operator_classname(self):
    self.assertEqual(AirflowDag._get_operator_classname('bash'), 'BashOperator')
    self.assertEqual(AirflowDag._get_operator_classname('bq'),
                     'BigQueryOperator')
    self.assertEqual(AirflowDag._get_operator_classname('Unknown'),
                     'UnknownOperator')

  def test_get_operator_param_name(self):
    self.assertEqual(AirflowDag._get_operator_param_name('query', 'bq'),
                     'bql')

  def test_get_dag_definition(self):
    self.assertEqual(AirflowDag._get_dag_definition('foo', 'bar'),
                     'dag = DAG(dag_id=\'foo\', schedule_interval=\'bar\', ' \
                     'default_args=default_args)\n\n')

  def test_default_args(self):
    self.assertEqual(
        AirflowDag._default_args_format.format(
            'foo@bar.com', 'Jun 1 2005  1:33PM', 'Jun 10 2005  1:33PM',
            '%b %d %Y %I:%M%p'), """
    'owner': 'Datalab',
    'depends_on_past': False,
    'email': ['foo@bar.com'],
    'start_date': datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p'),
    'end_date': datetime.strptime('Jun 10 2005  1:33PM', '%b %d %Y %I:%M%p'),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
""")

  def test_py(self):
    dag_spec = """
email: foo@bar.com
start_date: Jun 1 2005  1:33PM
end_date: Jun 10 2005  1:33PM
datetime_format: '%b %d %Y %I:%M%p'
dag_id: test_dag
schedule_interval: '@hourly'
tasks:
  print_pdt_date:
    type: bash
    bash_command: date
  print_utc_date:
    type: bash
    bash_command: date -u
    up_stream:
      - print_pdt_date
"""
    airflow_dag = AirflowDag(dag_spec)
    py_string = airflow_dag.py()
    self.assertEqual(py_string, """
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.contrib.operators.bigquery_operator import BigQueryOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'Datalab',
    'depends_on_past': False,
    'email': ['foo@bar.com'],
    'start_date': datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p'),
    'end_date': datetime.strptime('Jun 10 2005  1:33PM', '%b %d %Y %I:%M%p'),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(dag_id='test_dag', schedule_interval='@hourly', default_args=default_args)

print_utc_date = BashOperator(task_id='print_utc_date_id', bash_command='date -u', dag=dag)
print_pdt_date = BashOperator(task_id='print_pdt_date_id', bash_command='date', dag=dag)
print_utc_date.set_upstream(print_pdt_date)
""")

  def test_py_bq(self):
    dag_spec = """
email: rajivpb@google.com
start_date: Jun 21 2017  1:00AM
end_date: Jun 25 2017  1:33PM
datetime_format: '%b %d %Y %I:%M%p'
dag_id: demo_bq_dag_during_demo
schedule_interval: '0-59 * * * *'
tasks:
  current_timestamp:
    type: bq
    bql: INSERT INTO rajivpb_demo.the_datetime_table (the_datetime) VALUES (CURRENT_DATETIME())
    use_legacy_sql: False
  tomorrows_timestamp:
    type: bq
    bql: INSERT INTO rajivpb_demo.the_datetime_table (the_datetime) VALUES (CURRENT_DATETIME())
    use_legacy_sql: False
    up_stream:
      - current_timestamp
"""
    airflow_dag = AirflowDag(dag_spec)
    py_string = airflow_dag.py()

    # Print it to a file
    dat_py_file = open("demo_bq_dag_during_demo.py", "w")
    dat_py_file.write(py_string)
    dat_py_file.close()

    # Kick off gsutil in subprocess
    import subprocess
    subprocess.check_call(['gsutil', '-q', 'cp',
                           '/usr/local/google/home/rajivpb/IdeaProjects/pydatalab/demo_bq_dag_during_demo.py',
                           'gs://airflow-staging-test36490808-bucket/dags'])
    self.assertEqual(py_string, '')


if __name__ == '__main__':
  unittest.main()
