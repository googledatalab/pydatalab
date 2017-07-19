import google.datalab.utils._coders

from enum import Enum

class AirflowOperator(Enum):
  BigQuery = 'bq'
  Bash = 'bash'

class AirflowDag(object):
  """A coder to encode and decode CloudML metadata."""

  _imports = """
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.contrib.operators.bigquery_operator import BigQueryOperator
from datetime import datetime, timedelta

"""

  _default_args_format = """
    'owner': 'Datalab',
    'depends_on_past': False,
    'email': ['{0}'],
    'start_date': datetime.strptime('{1}', '{3}'),
    'end_date': datetime.strptime('{2}', '{3}'),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
"""

  def __init__(self, spec_str):
    self._spec_str = spec_str

  @property
  def spec(self):
    return self._spec_str

  def py(self):
    if not self._spec_str:
      return None

    dag_spec = self._decode(self._spec_str)
    default_args = 'default_args = {' + \
                   AirflowDag._default_args_format.format(
                       dag_spec['email'], dag_spec['start_date'],
                       dag_spec['end_date'],
                       dag_spec['datetime_format']) + '}\n\n'

    dag_definition = self._get_dag_definition(
        dag_spec['dag_id'], dag_spec['schedule_interval'])

    task_definitions = ''
    up_steam_statements = ''
    for task_id, task_details in dag_spec['tasks'].iteritems():
      task_def = AirflowDag._get_operator_definition(task_id, task_details)
      task_definitions = task_definitions + task_def
      dependency_def = \
        AirflowDag._get_dependency_definition(task_id,
                                              task_details.get('up_stream', []))
      up_steam_statements = up_steam_statements + dependency_def

    return AirflowDag._imports + \
           default_args + \
           dag_definition + \
           task_definitions + \
           up_steam_statements

  @staticmethod
  def _decode(content_string):
    try:
      return google.datalab.utils._coders.JsonCoder().decode(content_string)
    except ValueError:
      return google.datalab.utils._coders.YamlCoder().decode(content_string)

  @staticmethod
  def _get_operator_definition(task_id, task_details):
    operator_classname = AirflowDag._get_operator_classname(
        task_details['type'])
    param_string = 'task_id=\'{0}_id\''.format(task_id)
    for param_name, param_value in task_details.iteritems():
      # These are special-types that are relevant to Datalab
      if param_name in  ['type', 'up_stream']:
        continue
      # If the type is a python non-string (best guess), we don't quote it.
      if type(param_value) in [int, bool, float, type(None)]:
        param_format_string = ', {0}={1}'
      else:
        param_format_string = ', {0}=\'{1}\''
      param_string = param_string + param_format_string.format(param_name,
                                                               param_value)
    return '{0} = {1}({2}, dag=dag)\n'.format(
        task_id,
        operator_classname,
        param_string)

  @staticmethod
  def _get_dag_definition(dag_id, schedule_interval):
    dag_definition = 'dag = DAG(dag_id=\'{0}\', schedule_interval=\'{1}\', ' \
                     'default_args=default_args)\n\n'.format(dag_id,
                                                             schedule_interval)
    return dag_definition

  @staticmethod
  def _get_dependency_definition(task_id, dependencies):
    set_upstream_statements = ''
    for dependency in dependencies:
      set_upstream_statements = set_upstream_statements + \
                                '{0}.set_upstream({1})'.format(task_id,
                                                               dependency) + \
                                '\n'
    return set_upstream_statements

  @staticmethod
  def _get_operator_classname(task_detail_type):
    try:
      operator_enum = AirflowOperator(task_detail_type).name
    except ValueError:
      operator_enum = task_detail_type
    operator_classname = '{0}Operator'.format(operator_enum)
    return operator_classname
