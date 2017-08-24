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

from google.datalab import utils
from enum import Enum


class Operator(Enum):
  """ Represents a mapping from the Airflow operator class name suffix (i.e. the
  portion of the class name before the "Operator", and the corresponding string
  used in the yaml config (in the cell-body of '%pipeline create'). This
  mapping enables us to onboard additional Airflow operators with minimal code
  changes.
  """
  BigQuery = 'bq'
  BigQueryTableDelete = 'bq-table-delete'
  BigQueryToBigQuery = 'bq-to-bq'
  BigQueryToCloudStorage = 'bq-to-gcs'
  Bash = 'bash'


class Pipeline(object):
  """ Represents a Pipeline object that encapsulates an Airflow pipeline spec.

  This object can be used to generate the python airflow spec.
  """

  _imports = """
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.contrib.operators.bigquery_operator import BigQueryOperator
from airflow.contrib.operators.bigquery_table_delete_operator import BigQueryTableDeleteOperator
from airflow.contrib.operators.bigquery_to_bigquery import BigQueryToBigQueryOperator
from airflow.contrib.operators.bigquery_to_gcs import BigQueryToCloudStorageOperator
from datetime import timedelta
from pytz import timezone

"""

  _default_args_format = """
    'owner': 'Datalab',
    'depends_on_past': False,
    'email': ['{0}'],
    'start_date': {1},
    'end_date': {2},
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
"""

  def __init__(self, spec_str, name, env=None):
    """ Initializes an instance of a Pipeline object.

    Args:
      spec_str: the yaml config (cell body) of the pipeline from Datalab
      name: name of the pipeline (line argument) from Datalab
      env: a dictionary containing objects from the pipeline execution context,
          used to get references to Bigquery SQL objects, and other python
          objects defined in the notebook.
    """
    self._spec_str = spec_str
    self._env = env or {}
    self._name = name

  @property
  def py(self):
    """ Gets the airflow python spec for the Pipeline object. This is the
      input for the Cloud Composer service.
    """
    if not self._spec_str:
      return None

    dag_spec = utils.commands.parse_config(
        self._spec_str, self._env)

    start_date = dag_spec.get('schedule').get('start_date')
    end_date = dag_spec.get('schedule').get('end_date')
    default_args = 'default_args = {' + \
                   Pipeline._default_args_format.format(
                       dag_spec['email'],
                       self._get_datetime_expr_str(start_date),
                       self._get_datetime_expr_str(end_date)) + \
                   '}\n\n'

    dag_definition = self._get_dag_definition(
        dag_spec.get('schedule')['schedule_interval'])

    task_definitions = ''
    up_steam_statements = ''
    for (task_id, task_details) in sorted(dag_spec['tasks'].items()):
      task_def = self._get_operator_definition(task_id, task_details)
      task_definitions = task_definitions + task_def
      dependency_def = Pipeline._get_dependency_definition(
          task_id, task_details.get('up_stream', []))
      up_steam_statements = up_steam_statements + dependency_def

    return Pipeline._imports + default_args + dag_definition + \
        task_definitions + up_steam_statements

  @staticmethod
  def _get_datetime_expr_str(date_string):
    expr_format = 'datetime.datetime.strptime(\'{1}\', \'{2}\').replace(tzinfo=timezone(\'UTC\'))'
    datetime_format = '%Y-%m-%dT%H:%M:%SZ'  # ISO 8601, always UTC
    return expr_format.format(date_string, date_string, datetime_format)

  def _get_operator_definition(self, task_id, task_details):
    """ Internal helper that gets the Airflow operator for the task with the
      python parameters.
    """
    operator_type = task_details['type']
    param_string = 'task_id=\'{0}_id\''.format(task_id)
    Pipeline._add_default_override_params(task_details, operator_type)

    for (param_name, param_value) in sorted(task_details.items()):
      # These are special-types that are relevant to Datalab
      if param_name in ['type', 'up_stream']:
        continue
      operator_param_name = Pipeline._get_operator_param_name(param_name,
                                                              operator_type)
      operator_param_value = self._get_operator_param_value(
          param_name, operator_type, param_value)
      param_format_string = Pipeline._get_param_format_string(param_value)
      param_string = param_string + param_format_string.format(
          operator_param_name, operator_param_value)

    operator_classname = Pipeline._get_operator_classname(operator_type)

    return '{0} = {1}({2}, dag=dag)\n'.format(
        task_id,
        operator_classname,
        param_string)

  @staticmethod
  def _get_param_format_string(param_value):
    # If the type is a python non-string (best guess), we don't quote it.
    if type(param_value) in [int, bool, float, type(None)]:
      return ', {0}={1}'
    return ', {0}=\'{1}\''

  def _get_dag_definition(self, schedule_interval):
    dag_definition = 'dag = DAG(dag_id=\'{0}\', schedule_interval=\'{1}\', ' \
                     'default_args=default_args)\n\n'.format(self._name,
                                                             schedule_interval)
    return dag_definition

  @staticmethod
  def _get_dependency_definition(task_id, dependencies):
    """ Internal helper collects all the dependencies of the task, and returns
      the Airflow equivalent python sytax for specifying them.
    """
    set_upstream_statements = ''
    for dependency in dependencies:
      set_upstream_statements = set_upstream_statements + \
          '{0}.set_upstream({1})'.format(task_id, dependency) + '\n'
    return set_upstream_statements

  @staticmethod
  def _get_operator_classname(task_detail_type):
    """ Internal helper gets the name of the Airflow operator class. We maintain
      this in an enum, so this method really returns the enum name, concatenated
      with the string "Operator".
    """
    try:
      operator_enum = Operator(task_detail_type).name
    except ValueError:
      operator_enum = task_detail_type
    operator_classname = '{0}Operator'.format(operator_enum)
    return operator_classname

  @staticmethod
  def _get_operator_param_name(param_name, operator_type):
    """ Internal helper gets the name of the python parameter for the Airflow
      operator class. In some cases, we do not expose the airflow parameter
      name in its native form, but choose to couch it with a name that's more
      Datalab friendly. For example, Airflow's BigQueryOperator uses 'bql' for
      the query string, but we have chosen to expose this as 'query' to the
      Datalab user. Hence, a few substitutions that are specific to the operator
      type need to be made.
    """
    if (operator_type == 'bq'):
      if (param_name == 'query'):
        return 'bql'
    return param_name

  def _get_operator_param_value(self, param_name, operator_type, param_value):
    """ Internal helper gets the python parameter value for the Airflow
      operator class. It needs to make exceptions that are specific to the
      operator-type, in some ways similar to _get_operator_param_name.
    """
    if (operator_type == 'bq') and (param_name in ['query', 'bql']):
        return param_value.sql
    return param_value

  @staticmethod
  def _add_default_override_params(task_details, operator_type):
    """ Internal helper that overrides the defaults of an Airflow operator's
      parameters, when necessary.
    """
    if operator_type == 'bq':
      bq_defaults = {}
      bq_defaults['use_legacy_sql'] = False
      for param_name, param_value in bq_defaults.items():
          if param_name not in task_details:
            task_details[param_name] = param_value
