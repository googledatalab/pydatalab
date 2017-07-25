import google.datalab.utils._coders
import datetime

from enum import Enum

class AirflowOperator(Enum):
  BigQuery = 'bq'
  Bash = 'bash'

class AirflowPipeline(object):
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
                   AirflowPipeline._default_args_format.format(
                       dag_spec['email'],
                       dag_spec.get('schedule').get('start_date'),
                       dag_spec.get('schedule').get('end_date'),
                       dag_spec.get('schedule').get('datetime_format')) + '}\n\n'

    dag_definition = self._get_dag_definition(
        dag_spec['pipeline_id'], dag_spec.get('schedule')['schedule_interval'])

    task_definitions = ''
    up_steam_statements = ''
    for task_id, task_details in dag_spec['tasks'].iteritems():
      task_def = AirflowPipeline._get_operator_definition(task_id, task_details)
      task_definitions = task_definitions + task_def
      dependency_def = \
        AirflowPipeline._get_dependency_definition(task_id,
                                              task_details.get('up_stream', []))
      up_steam_statements = up_steam_statements + dependency_def

    return AirflowPipeline._imports + \
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
    operator_type = task_details['type']
    param_string = 'task_id=\'{0}_id\''.format(task_id)
    AirflowPipeline._add_default_params(task_details, operator_type)

    for param_name, param_value in task_details.iteritems():
      # These are special-types that are relevant to Datalab
      if param_name in  ['type', 'up_stream']:
        continue
      # If the type is a python non-string (best guess), we don't quote it.
      if type(param_value) in [int, bool, float, type(None)]:
        param_format_string = ', {0}={1}'
      else:
        param_format_string = ', {0}=\'{1}\''
      operator_param_name = AirflowPipeline._get_operator_param_name(param_name,
                                                                operator_type)
      param_string = param_string + param_format_string.format(
          operator_param_name, param_value)

    operator_classname = AirflowPipeline._get_operator_classname(operator_type)

    return '{0} = {1}({2}, dag=dag)\n'.format(
        task_id,
        operator_classname,
        param_string)

  @staticmethod
  def _get_dag_definition(pipeline_id, schedule_interval):
    dag_definition = 'dag = DAG(dag_id=\'{0}\', schedule_interval=\'{1}\', ' \
                     'default_args=default_args)\n\n'.format(pipeline_id,
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

  @staticmethod
  def _get_operator_param_name(param_name, operator_type):
    if (operator_type == 'bq'):
      if (param_name == 'query'):
        return 'bql'
    return param_name

  @staticmethod
  def _add_default_params(task_details, operator_type):
    if operator_type == 'bq':
      bq_defaults = {}
      bq_defaults['destination_dataset_table'] = False
      bq_defaults['write_disposition'] = 'WRITE_EMPTY'
      bq_defaults['allow_large_results'] = False
      bq_defaults['bigquery_conn_id'] = 'bigquery_default'
      bq_defaults['delegate_to'] = None
      bq_defaults['udf_config'] = False
      bq_defaults['use_legacy_sql'] = False
      for param_name, param_value in bq_defaults.iteritems():
          if param_name not in task_details:
            task_details[param_name] = param_value
