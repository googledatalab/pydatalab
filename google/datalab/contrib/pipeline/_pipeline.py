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

import datetime
import google
import google.datalab.bigquery as bigquery
import sys

# Any operators need to be imported here. This is required for dynamically getting the list of
# templated fields from the operators. Static code-analysis will report that this is not
# necessary, hence the '# noqa' annotations
from google.datalab.contrib.bigquery.operators._bq_load_operator import LoadOperator  # noqa
from google.datalab.contrib.bigquery.operators._bq_execute_operator import ExecuteOperator  # noqa
from google.datalab.contrib.bigquery.operators._bq_extract_operator import ExtractOperator  # noqa


class PipelineGenerator(object):
  """ Represents a Pipeline object that encapsulates an Airflow pipeline spec.

  This object can be used to generate the python airflow spec.
  """

  _imports = """
import datetime
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.contrib.operators.bigquery_operator import BigQueryOperator
from airflow.contrib.operators.bigquery_table_delete_operator import BigQueryTableDeleteOperator
from airflow.contrib.operators.bigquery_to_bigquery import BigQueryToBigQueryOperator
from airflow.contrib.operators.bigquery_to_gcs import BigQueryToCloudStorageOperator
from airflow.contrib.operators.gcs_to_bq import GoogleCloudStorageToBigQueryOperator
from google.datalab.contrib.bigquery.operators._bq_load_operator import LoadOperator
from google.datalab.contrib.bigquery.operators._bq_execute_operator import ExecuteOperator
from google.datalab.contrib.bigquery.operators._bq_extract_operator import ExtractOperator
from datetime import timedelta
"""

  @staticmethod
  def generate_airflow_spec(name, pipeline_spec):
    """ Gets the airflow python spec for the Pipeline object.
    """
    task_definitions = ''
    up_steam_statements = ''
    parameters = pipeline_spec.get('parameters')
    for (task_id, task_details) in sorted(pipeline_spec['tasks'].items()):
      task_def = PipelineGenerator._get_operator_definition(task_id, task_details, parameters)
      task_definitions = task_definitions + task_def
      dependency_def = PipelineGenerator._get_dependency_definition(
        task_id, task_details.get('up_stream', []))
      up_steam_statements = up_steam_statements + dependency_def

    schedule_config = pipeline_spec.get('schedule', {})

    default_args = PipelineGenerator._get_default_args(schedule_config,
                                                       pipeline_spec.get('emails', {}))
    dag_definition = PipelineGenerator._get_dag_definition(
      name, schedule_config.get('interval', '@once'), schedule_config.get('catchup', False))
    return PipelineGenerator._imports + default_args + dag_definition + task_definitions + \
        up_steam_statements

  @staticmethod
  def _get_default_args(schedule_config, emails):
    start_datetime_obj = schedule_config.get('start', datetime.datetime.now())
    end_datetime_obj = schedule_config.get('end')
    start_date_str = PipelineGenerator._get_datetime_expr_str(start_datetime_obj)
    end_date_str = PipelineGenerator._get_datetime_expr_str(end_datetime_obj)

    default_arg_literals = """
    'owner': 'Google Cloud Datalab',
    'email': {0},
    'start_date': {1},
    'end_date': {2},
""".format(emails.split(',') if emails else [], start_date_str, end_date_str)

    configurable_keys = ['email_on_retry', 'email_on_failure', 'retries',
                         'retry_exponential_backoff']
    for configurable_key in configurable_keys:
      if configurable_key in schedule_config:
        default_arg_literals = default_arg_literals + """    \'{0}\': {1},
""".format(configurable_key, schedule_config.get(configurable_key))

    # We deal with these separately as they need to be timedelta literals.
    retry_delay_keys = ['retry_delay_seconds', 'max_retry_delay_seconds']
    for retry_delay_key in retry_delay_keys:
      if retry_delay_key in schedule_config:
        default_arg_literals = default_arg_literals + """    \'{0}\': timedelta(seconds={1}),
""".format(retry_delay_key[:-8], schedule_config.get(retry_delay_key))

    return """
default_args = {{{0}}}

""".format(default_arg_literals)

  @staticmethod
  def _get_datetime_expr_str(datetime_obj):
    if not datetime_obj:
      return None

    # Apache Airflow assumes that all times are timezone-unaware, and are in UTC:
    # https: // issues.apache.org / jira / browse / AIRFLOW - 1710
    # Somewhat conveniently, yaml.load() recognizes and parses strings that look like datetimes
    # into timezone unaware datetime objects (if the user input specifies the timezone, it's
    # corrected and the result is assumed to be in UTC).
    # Here, we serialize this object into the format laid down by ISO 8601, and generate python code
    # that parses this format into a datetime object for Airflow.
    datetime_format = '%Y-%m-%dT%H:%M:%S'  # ISO 8601, timezone unaware
    expr_format = 'datetime.datetime.strptime(\'{0}\', \'{1}\')'
    return expr_format.format(datetime_obj.strftime(datetime_format), datetime_format)

  @staticmethod
  def _get_operator_definition(task_id, task_details, parameters):
    """ Internal helper that gets the definition of the airflow operator for the task with the
      python parameters. All the parameters are also expanded with the airflow macros.
      :param parameters:
    """
    operator_type = task_details['type']
    full_param_string = 'task_id=\'{0}_id\''.format(task_id)
    operator_class_name, module = PipelineGenerator._get_operator_class_name(operator_type)
    operator_class_instance = getattr(sys.modules[module], operator_class_name, None)
    templated_fields = operator_class_instance.template_fields if operator_class_instance else ()

    operator_param_values = PipelineGenerator._get_operator_param_name_and_values(
        operator_class_name, task_details)

    # This loop resolves all the macros and builds up the final string
    merged_parameters = google.datalab.bigquery.Query.merge_parameters(
      parameters, date_time=datetime.datetime.now(), macros=True, types_and_values=False)
    for (operator_param_name, operator_param_value) in sorted(operator_param_values.items()):
      # We replace modifiers in the parameter values with either the user-defined values, or with
      # with the airflow macros, as applicable.
      # An important assumption that this makes is that the operators parameters have the same names
      # as the templated_fields. TODO(rajivpb): There may be a better way to do this.
      if operator_param_name in templated_fields:
        operator_param_value = google.datalab.bigquery.Query._resolve_parameters(
          operator_param_value, merged_parameters)
      param_format_string = PipelineGenerator._get_param_format_string(operator_param_value)
      param_string = param_format_string.format(operator_param_name, operator_param_value)
      full_param_string = full_param_string + param_string

    return '{0} = {1}({2}, dag=dag)\n'.format(task_id, operator_class_name, full_param_string)

  @staticmethod
  def _get_param_format_string(param_value):
    # If the type is a python non-string (best guess), we don't quote it.
    if type(param_value) in [int, bool, float, type(None), list, dict]:
      return ', {0}={1}'
    return ', {0}="""{1}"""'

  @staticmethod
  def _get_dag_definition(name, schedule_interval, catchup=False):
    dag_definition = 'dag = DAG(dag_id=\'{0}\', schedule_interval=\'{1}\', ' \
                     'catchup={2}, default_args=default_args)\n\n'.format(name,
                                                                          schedule_interval,
                                                                          catchup)
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
  def _get_operator_class_name(task_detail_type):
    """ Internal helper gets the name of the Airflow operator class. We maintain
      this in a map, so this method really returns the enum name, concatenated
      with the string "Operator".
    """
    # TODO(rajivpb): Rename this var correctly.
    task_type_to_operator_prefix_mapping = {
      'pydatalab.bq.execute': ('Execute',
                               'google.datalab.contrib.bigquery.operators._bq_execute_operator'),
      'pydatalab.bq.extract': ('Extract',
                               'google.datalab.contrib.bigquery.operators._bq_extract_operator'),
      'pydatalab.bq.load': ('Load', 'google.datalab.contrib.bigquery.operators._bq_load_operator'),
      'Bash': ('Bash', 'airflow.operators.bash_operator')
    }
    (operator_class_prefix, module) = task_type_to_operator_prefix_mapping.get(
        task_detail_type, (None, __name__))
    format_string = '{0}Operator'
    operator_class_name = format_string.format(operator_class_prefix)
    if operator_class_prefix is None:
      return format_string.format(task_detail_type), module
    return operator_class_name, module

  @staticmethod
  def _get_operator_param_name_and_values(operator_class_name, task_details):
    """ Internal helper gets the name of the python parameter for the Airflow operator class. In
      some cases, we do not expose the airflow parameter name in its native form, but choose to
      expose a name that's more standard for Datalab, or one that's more friendly. For example,
      Airflow's BigQueryOperator uses 'bql' for the query string, but we want %%bq users in Datalab
      to use 'query'. Hence, a few substitutions that are specific to the Airflow operator need to
      be made.

      Similarly, we the parameter value could come from the notebook's context. All that happens
      here.

      Returns:
        Dict containing _only_ the keys and values that are required in Airflow operator definition.
      This requires a substituting existing keys in the dictionary with their Airflow equivalents (
      i.e. by adding new keys, and removing the existing ones).
    """

    # We make a clone and then remove 'type' and 'up_stream' since these aren't needed for the
    # the operator's parameters.
    operator_task_details = task_details.copy()
    if 'type' in operator_task_details.keys():
      del operator_task_details['type']
    if 'up_stream' in operator_task_details.keys():
      del operator_task_details['up_stream']

    # We special-case certain operators if we do some translation of the parameter names. This is
    # usually the case when we use syntactic sugar to expose the functionality.
    # TODO(rajivpb): It should be possible to make this a lookup from the modules mapping via
    # getattr() or equivalent. Avoid hard-coding these class-names here.
    if (operator_class_name == 'BigQueryOperator'):
      return PipelineGenerator._get_bq_execute_params(operator_task_details)
    if (operator_class_name == 'BigQueryToCloudStorageOperator'):
      return PipelineGenerator._get_bq_extract_params(operator_task_details)
    if (operator_class_name == 'GoogleCloudStorageToBigQueryOperator'):
      return PipelineGenerator._get_bq_load_params(operator_task_details)
    return operator_task_details

  @staticmethod
  def _get_bq_execute_params(operator_task_details):
    if 'query' in operator_task_details:
      operator_task_details['bql'] = operator_task_details['query'].sql
      del operator_task_details['query']

    if 'parameters' in operator_task_details:
      operator_task_details['query_params'] = bigquery.Query.get_query_parameters(
        operator_task_details['parameters'])
      del operator_task_details['parameters']

    # Add over-rides of Airflow defaults here.
    if 'use_legacy_sql' not in operator_task_details:
      operator_task_details['use_legacy_sql'] = False

    return operator_task_details

  @staticmethod
  def _get_bq_extract_params(operator_task_details):
    if 'table' in operator_task_details:
      table = bigquery.commands._bigquery._get_table(operator_task_details['table'])
      operator_task_details['source_project_dataset_table'] = table.full_name
      del operator_task_details['table']
    if 'path' in operator_task_details:
      operator_task_details['destination_cloud_storage_uris'] = [operator_task_details['path']]
      del operator_task_details['path']
    if 'format' in operator_task_details:
      operator_task_details['export_format'] = 'CSV' if operator_task_details['format'] == 'csv' \
        else 'NEWLINE_DELIMITED_JSON'
      del operator_task_details['format']
    if 'delimiter' in operator_task_details:
      operator_task_details['field_delimiter'] = operator_task_details['delimiter']
      del operator_task_details['delimiter']
    if 'compress' in operator_task_details:
      operator_task_details['compression'] = 'GZIP' if operator_task_details['compress'] else 'NONE'
      del operator_task_details['compress']
    if 'header' in operator_task_details:
      operator_task_details['print_header'] = operator_task_details['header']
      del operator_task_details['header']

    return operator_task_details

  @staticmethod
  def _get_bq_load_params(operator_task_details):
    if 'table' in operator_task_details:
      table = bigquery.commands._bigquery._get_table(operator_task_details['table'])
      if not table:
        table = bigquery.Table(operator_task_details['table'])
        # TODO(rajivpb): Ensure that mode == create here.
      operator_task_details['destination_project_dataset_table'] = table.full_name
      del operator_task_details['table']

    if 'format' in operator_task_details:
      operator_task_details['export_format'] = 'CSV' if operator_task_details['format'] == 'csv' \
        else 'NEWLINE_DELIMITED_JSON'
      del operator_task_details['format']

    if 'delimiter' in operator_task_details:
      operator_task_details['field_delimiter'] = operator_task_details['delimiter']
      del operator_task_details['delimiter']

    if 'skip' in operator_task_details:
      operator_task_details['skip_leading_rows'] = operator_task_details['skip']
      del operator_task_details['skip']

    if 'path' in operator_task_details:
      bucket, source_object = PipelineGenerator._get_bucket_and_source_object(
        operator_task_details['path'])
      operator_task_details['bucket'] = bucket
      operator_task_details['source_objects'] = source_object
      del operator_task_details['path']

    return operator_task_details

  @staticmethod
  def _get_bucket_and_source_object(gcs_path):
    return gcs_path.split('/')[2], '/'.join(gcs_path.split('/')[3:])
