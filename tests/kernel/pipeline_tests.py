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
import mock
from oauth2client.client import AccessTokenCredentials
import unittest
import json
import pandas
from datetime import datetime
try:
  from StringIO import StringIO
except ImportError:
  from io import StringIO

# import Python so we can mock the parts we need to here.
import IPython
import IPython.core.magic


def noop_decorator(func):
  return func


IPython.core.magic.register_line_cell_magic = noop_decorator
IPython.core.magic.register_line_magic = noop_decorator
IPython.core.magic.register_cell_magic = noop_decorator
IPython.get_ipython = mock.Mock()


import google.datalab  # noqa
import google.datalab.airflow  # noqa
import google.datalab.airflow.commands  # noqa
import google.datalab.utils.commands  # noqa


class TestCases(unittest.TestCase):

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = AccessTokenCredentials('test_token', 'test_ua')
    return google.datalab.Context(project_id, creds)

  @mock.patch('google.datalab.airflow.Pipeline.execute')
  @mock.patch('google.datalab.utils.commands.notebook_environment')
  @mock.patch('google.datalab.Context.default')
  def test_create_cell(self, mock_default_context, mock_notebook_environment, mock_create_execute):
    env = {}
    mock_default_context.return_value = TestCases._create_context()
    mock_notebook_environment.return_value = env
    IPython.get_ipython().user_ns = env

    # test pipeline creation
    p_body = """
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

    # no pipeline name specified. should execute
    google.datalab.airflow.commands._pipeline._create_cell({'name': None}, p_body)
    mock_create_execute.assert_called_with()

    # test pipeline creation
    google.datalab.airflow.commands._pipeline._create_cell({'name': 'p1'}, p_body)
    p1 = env['p1']
    self.assertIsNotNone(p1)
    self.assertEqual(p_body, p1.spec)
    self.assertEqual("""
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
""", p1.py)
