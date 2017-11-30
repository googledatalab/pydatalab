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
import google.datalab.bigquery as bigquery
from google.datalab import utils
import six


class Pipeline(object):
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

  # These are documented here:
  # https://airflow.incubator.apache.org/code.html?highlight=macros#default-variables
  airflow_macros = {
    # the datetime formatted as YYYY-MM-DD
    '_ds': '{{ ds }}',
    # the full ISO-formatted timestamp YYYY-MM-DDTHH:MM:SS.mmmmmm
    '_ts': '{{ ts }}',
    # the datetime formatted as YYYYMMDD (i.e. YYYY-MM-DD with 'no dashes')
    '_ds_nodash': '{{ ds_nodash }}',
    # the timestamp formatted as YYYYMMDDTHHMMSSmmmmmm (i.e full ISO-formatted timestamp
    # YYYY-MM-DDTHH:MM:SS.mmmmmm with no dashes or colons).
    '_ts_nodash': '{{ ts_nodash }}',
    '_ts_year': "{{ execution_date.year }}",
    '_ts_month': "{{ execution_date.month }}",
    '_ts_day': "{{ execution_date.day }}",
    '_ts_hour': "{{ execution_date.hour }}",
    '_ts_minute': "{{ execution_date.minute }}",
    '_ts_second': "{{ execution_date.second }}",
  }

  def __init__(self, name, pipeline_spec, resolve_airflow_macros=False):
    """ Initializes an instance of a Pipeline object.

    Args:
      pipeline_spec: Dict with pipeline-spec in key-value form.
    """
    self._pipeline_spec = pipeline_spec
    self._name = name
    self._resolve_airflow_macros = resolve_airflow_macros

  def get_airflow_spec(self):
    """ Gets the airflow python spec (Composer service input) for the Pipeline object.
    """
    task_definitions = ''
    up_steam_statements = ''
    parameters = self._pipeline_spec.get('parameters')
    for (task_id, task_details) in sorted(self._pipeline_spec['tasks'].items()):
      task_def = self._get_operator_definition(task_id, task_details, parameters)
      task_definitions = task_definitions + task_def
      dependency_def = Pipeline._get_dependency_definition(task_id, task_details.get('up_stream',
                                                                                     []))
      up_steam_statements = up_steam_statements + dependency_def

    schedule_config = self._pipeline_spec.get('schedule', {})

    default_args = Pipeline._get_default_args(schedule_config,
                                              self._pipeline_spec.get('emails', {}))
    dag_definition = self._get_dag_definition(schedule_config.get('interval', '@once'))
    self._airflow_spec = Pipeline._imports + default_args + dag_definition + task_definitions + \
        up_steam_statements
    return self._airflow_spec

  @staticmethod
  def get_pipeline_spec(spec_str, env=None):
    """ Gets a dict representation of the pipeline-spec, given a yaml string.
    Args:
      spec_str: string representation of the pipeline's yaml spec
      env: a dictionary containing objects from the pipeline execution context,
          used to get references to Bigquery SQL objects, and other python
          objects defined in the notebook.

    Returns:
      Dict with pipeline-spec in key-value form.
    """
    if not spec_str:
      return None
    return utils.commands.parse_config(spec_str, env)

  @staticmethod
  def _get_default_args(schedule_config, emails):
    start_datetime_obj = schedule_config.get('start', datetime.datetime.now())
    end_datetime_obj = schedule_config.get('end')
    start_date_str = Pipeline._get_datetime_expr_str(start_datetime_obj)
    end_date_str = Pipeline._get_datetime_expr_str(end_datetime_obj)

    airflow_default_args_format = """
default_args = {{
    'owner': 'Google Cloud Datalab',
    'email': {2},
    'start_date': {0},
    'end_date': {1},
}}

"""
    email_list = emails.split(',') if emails else []
    return airflow_default_args_format.format(start_date_str, end_date_str, email_list)

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

  def _get_operator_definition(self, task_id, task_details, parameters):
    """ Internal helper that gets the definition of the airflow operator for the task with the
      python parameters. All the parameters are also expanded with the airflow macros.
      :param parameters:
    """
    operator_type = task_details['type']
    full_param_string = 'task_id=\'{0}_id\''.format(task_id)
    operator_classname = Pipeline._get_operator_classname(operator_type)

    operator_param_values = Pipeline._get_operator_param_name_and_values(
        operator_classname, task_details)
    for (operator_param_name, operator_param_value) in sorted(operator_param_values.items()):
      if self._resolve_airflow_macros:
        operator_param_value = self._resolve_parameters(operator_param_value,
                                                        Pipeline.merge_parameters(parameters))
      param_format_string = Pipeline._get_param_format_string(
          operator_param_value)
      param_string = param_format_string.format(operator_param_name, operator_param_value)
      full_param_string = full_param_string + param_string

    return '{0} = {1}({2}, dag=dag)\n'.format(task_id, operator_classname, full_param_string)

  @staticmethod
  def merge_parameters(parameters):
    # We merge the user-provided parameters and the airflow macros
    merged_parameters = Pipeline.airflow_macros.copy()
    # TODO(rajivpb): Ignoring 'type' for now; figure out how to use that later.
    if parameters:
      parameters_dict = {item['name']: item['value'] for item in parameters}
      merged_parameters.update(parameters_dict)

    return merged_parameters

  def _resolve_parameters(self, operator_param_value, merged_parameters):
    if isinstance(operator_param_value, list):
      return [self._resolve_parameters(item, merged_parameters) for item in operator_param_value]
    if isinstance(operator_param_value, dict):
      return {self._resolve_parameters(k, merged_parameters): self._resolve_parameters(
        v, merged_parameters) for k, v in operator_param_value.items()}
    if isinstance(operator_param_value, six.string_types) and merged_parameters:
      return operator_param_value % merged_parameters
    return operator_param_value

  @staticmethod
  def _get_param_format_string(param_value):
    # If the type is a python non-string (best guess), we don't quote it.
    if type(param_value) in [int, bool, float, type(None), list, dict]:
      return ', {0}={1}'
    return ', {0}="""{1}"""'

  def _get_dag_definition(self, schedule_interval):
    dag_definition = 'dag = DAG(dag_id=\'{0}\', schedule_interval=\'{1}\', ' \
                     'default_args=default_args)\n\n'.format(self._name, schedule_interval)
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
      this in a map, so this method really returns the enum name, concatenated
      with the string "Operator".
    """
    task_type_to_operator_prefix_mapping = {
      'pydatalab.bq.execute': 'Execute',
      'pydatalab.bq.extract': 'Extract',
      'pydatalab.bq.load': 'Load',
    }
    operator_class_prefix = task_type_to_operator_prefix_mapping.get(
        task_detail_type)
    format_string = '{0}Operator'
    if operator_class_prefix is not None:
      return format_string.format(operator_class_prefix)
    return format_string.format(task_detail_type)

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
    if (operator_class_name == 'BigQueryOperator'):
      return Pipeline._get_bq_execute_params(operator_task_details)
    if (operator_class_name == 'BigQueryToCloudStorageOperator'):
      return Pipeline._get_bq_extract_params(operator_task_details)
    if (operator_class_name == 'GoogleCloudStorageToBigQueryOperator'):
      return Pipeline._get_bq_load_params(operator_task_details)
    return operator_task_details

  @staticmethod
  def _get_bq_execute_params(operator_task_details):
    if 'query' in operator_task_details:
      operator_task_details['bql'] = operator_task_details['query'].sql
      del operator_task_details['query']

    if 'parameters' in operator_task_details:
      operator_task_details['query_params'] = Pipeline._get_query_parameters(
        operator_task_details['parameters'])
      del operator_task_details['parameters']

    # Add over-rides of Airflow defaults here.
    if 'use_legacy_sql' not in operator_task_details:
      operator_task_details['use_legacy_sql'] = False

    return operator_task_details

  @staticmethod
  def _get_query_parameters(input_query_parameters):
    """Extract query parameters from dict if provided
    Also validates the cell body schema using jsonschema to catch errors. This validation isn't
    complete, however; it does not validate recursive schemas, but it acts as a good filter
    against most simple schemas.

    Args:
      operator_task_details: dict of input param names to values.

    Returns:
      Validated object containing query parameters.
    """
    parsed_params = []
    if input_query_parameters:
      for param in input_query_parameters:
        parsed_params.append({
          'name': param['name'],
          'parameterType': {
            'type': param['type']
          },
          'parameterValue': {
            'value': param['value']
          }
        })
    return parsed_params

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
      bucket, source_object = Pipeline._get_bucket_and_source_object(operator_task_details['path'])
      operator_task_details['bucket'] = bucket
      operator_task_details['source_objects'] = source_object
      del operator_task_details['path']

    return operator_task_details

  @staticmethod
  def _get_bucket_and_source_object(gcs_path):
    return gcs_path.split('/')[2], '/'.join(gcs_path.split('/')[3:])
