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

from __future__ import absolute_import
from __future__ import unicode_literals

import unittest

# import Python so we can mock the parts we need to here.
import IPython
import IPython.core.magic
import mock
from oauth2client.client import AccessTokenCredentials

import google.datalab.contrib.pipeline.commands._pipeline


def noop_decorator(func):
  return func


IPython.core.magic.register_line_cell_magic = noop_decorator
IPython.core.magic.register_line_magic = noop_decorator
IPython.core.magic.register_cell_magic = noop_decorator
IPython.get_ipython = mock.Mock()


class TestCases(unittest.TestCase):

  # test pipeline creation
  sample_cell_body = """
email: foo@bar.com
schedule:
  start_date: 2009-05-05T22:28:15Z
  end_date: 2009-05-06T22:28:15Z
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

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = AccessTokenCredentials('test_token', 'test_ua')
    return google.datalab.Context(project_id, creds)

  @mock.patch('google.datalab.utils.commands.notebook_environment')
  @mock.patch('google.datalab.Context.default')
  def test_create_cell_no_name(self, mock_default_context,
                               mock_notebook_environment):
    env = {}
    mock_default_context.return_value = TestCases._create_context()
    mock_notebook_environment.return_value = env
    IPython.get_ipython().user_ns = env

    # test pipeline creation
    p_body = 'foo'

    # no pipeline name specified. should execute
    with self.assertRaises(Exception):
      google.datalab.contrib.pipeline.commands._pipeline._create_cell({'name': None},
                                                                      p_body)

  @mock.patch('google.datalab.utils.commands.notebook_environment')
  @mock.patch('google.datalab.Context.default')
  def test_create_cell_debug(self, mock_default_context,
                             mock_notebook_environment):
    env = {}
    mock_default_context.return_value = TestCases._create_context()
    mock_notebook_environment.return_value = env
    IPython.get_ipython().user_ns = env

    # cell output is empty when debug is True
    output = google.datalab.contrib.pipeline.commands._pipeline._create_cell(
        {'name': 'foo_pipeline', 'debug': True}, self.sample_cell_body)
    self.assertTrue(len(output) > 0)

    output = google.datalab.contrib.pipeline.commands._pipeline._create_cell(
        {'name': 'foo_pipeline', 'debug': False}, self.sample_cell_body)
    self.assertTrue(output is None)

    output = google.datalab.contrib.pipeline.commands._pipeline._create_cell(
        {'name': 'foo_pipeline'}, self.sample_cell_body)
    self.assertTrue(output is None)

  @mock.patch('google.datalab.utils.commands.notebook_environment')
  @mock.patch('google.datalab.Context.default')
  def test_create_cell(self, mock_default_context, mock_notebook_environment):
    env = {}
    mock_default_context.return_value = TestCases._create_context()
    mock_notebook_environment.return_value = env
    IPython.get_ipython().user_ns = env

    # test pipeline creation
    google.datalab.contrib.pipeline.commands._pipeline._create_cell(
        {'name': 'p1'}, self.sample_cell_body)

    p1 = env['p1']
    self.assertIsNotNone(p1)
    self.assertEqual(self.sample_cell_body, p1._spec_str)
    self.assertEqual(p1.py, """
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

dag = DAG(dag_id='p1', schedule_interval='@hourly', default_args=default_args)

print_pdt_date = BashOperator(task_id='print_pdt_date_id', bash_command='date', dag=dag)
print_utc_date = BashOperator(task_id='print_utc_date_id', bash_command='date -u', dag=dag)
print_utc_date.set_upstream(print_pdt_date)
"""  # noqa
    )

  @mock.patch('google.datalab.utils.commands.notebook_environment')
  @mock.patch('google.datalab.Context.default')
  def test_create_cell_with_variable(self, mock_default_context,
                                     mock_notebook_environment):
    mock_default_context.return_value = TestCases._create_context()
    env = {}
    env['foo_query'] = google.datalab.bigquery.Query(
        'SELECT * FROM publicdata.samples.wikipedia LIMIT 5')
    mock_notebook_environment.return_value = env
    # TODO(rajivpb): Possibly not necessary
    IPython.get_ipython().user_ns = env

    # test pipeline creation
    p_body = """
email: foo@bar.com
schedule:
  start_date: 2009-05-05T22:28:15Z
  end_date: 2009-05-06T22:28:15Z
  schedule_interval: '@hourly'
tasks:
  print_pdt_date:
    type: bq
    query: $foo_query
"""

    # no pipeline name specified. should execute
    google.datalab.contrib.pipeline.commands._pipeline._create_cell({'name': 'p1'}, p_body)
    p1 = env['p1']
    self.assertIsNotNone(p1)
    self.assertEqual(p_body, p1._spec_str)
    self.assertEqual(p1.py, """
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

dag = DAG(dag_id='p1', schedule_interval='@hourly', default_args=default_args)

print_pdt_date = BigQueryOperator(task_id='print_pdt_date_id', bql='SELECT * FROM publicdata.samples.wikipedia LIMIT 5', use_legacy_sql=False, dag=dag)
"""  # noqa
    )
