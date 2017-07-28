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
import unittest

import google.datalab
import google.datalab.airflow


class TestCases(unittest.TestCase):

  def test_create_pipeline(self):
    spec_str = """
pipeline_id: test_dag
email: foo@bar.com
schedule:
  start_date: Jun 1 2005  1:33PM
  end_date: Jun 10 2005  1:33PM
  datetime_format: '%b %d %Y %I:%M%p'
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
    p = TestCases._create_pipeline(spec_str, name='foo_pipeline')
    self.assertEqual(p.py, """
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.contrib.operators.bigquery_operator import BigQueryOperator
from airflow.contrib.operators.bigquery_table_delete_operator import BigQueryTableDeleteOperator
from airflow.contrib.operators.bigquery_to_bigquery import BigQueryToBigQueryOperator
from airflow.contrib.operators.bigquery_to_gcs import BigQueryToCloudStorageOperator
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

  @staticmethod
  def _create_pipeline(spec_str=None, name=None, env=None):
    if env is None:
      env = {}
    p = google.datalab.airflow.Pipeline(spec_str)
    if name:
      env[name] = p
    return p
