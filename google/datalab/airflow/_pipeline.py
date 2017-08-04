import google.datalab.utils._coders

from enum import Enum

class Operator(Enum):
  BigQuery = 'bq'
  BigQueryTableDelete = 'bq-table-delete'
  BigQueryToBigQuery = 'bq-to-bq'
  BigQueryToCloudStorage = 'bq-to-gcs'
  Bash = 'bash'

class Pipeline(object):

  _imports = """
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.contrib.operators.bigquery_operator import BigQueryOperator
from airflow.contrib.operators.bigquery_table_delete_operator import BigQueryTableDeleteOperator
from airflow.contrib.operators.bigquery_to_bigquery import BigQueryToBigQueryOperator
from airflow.contrib.operators.bigquery_to_gcs import BigQueryToCloudStorageOperator
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

  def __init__(self, spec_str, name=None, env=None):
    self._spec_str = spec_str
    self._env = env or {}
    self._name = name

  @property
  def spec(self):
    return self._spec_str

  @property
  def py(self):
    if not self._spec_str:
      return None

    dag_spec = self._decode(self._spec_str)
    default_args = 'default_args = {' + \
                   Pipeline._default_args_format.format(
                       dag_spec['email'],
                       dag_spec.get('schedule').get('start_date'),
                       dag_spec.get('schedule').get('end_date'),
                       dag_spec.get('schedule').get('datetime_format')) + '}\n\n'

    dag_definition = self._get_dag_definition(
        self._name, dag_spec.get('schedule')['schedule_interval'])

    task_definitions = ''
    up_steam_statements = ''
    for task_id, task_details in dag_spec['tasks'].iteritems():
      task_def = self._get_operator_definition(task_id, task_details)
      task_definitions = task_definitions + task_def
      dependency_def = \
        Pipeline._get_dependency_definition(task_id,
                                            task_details.get('up_stream', []))
      up_steam_statements = up_steam_statements + dependency_def

    return Pipeline._imports + \
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

  def _get_operator_definition(self, task_id, task_details):
    operator_type = task_details['type']
    param_string = 'task_id=\'{0}_id\''.format(task_id)
    Pipeline._add_default_params(task_details, operator_type)

    for param_name, param_value in task_details.iteritems():
      # These are special-types that are relevant to Datalab
      if param_name in  ['type', 'up_stream']:
        continue
      # If the type is a python non-string (best guess), we don't quote it.
      if type(param_value) in [int, bool, float, type(None)]:
        param_format_string = ', {0}={1}'
      else:
        param_format_string = ', {0}=\'{1}\''
      operator_param_name = Pipeline._get_operator_param_name(param_name,
                                                              operator_type)
      operator_param_value = self._get_operator_param_value(param_name,
                                                                operator_type,
                                                                param_value)
      param_string = param_string + param_format_string.format(
          operator_param_name, operator_param_value)

    operator_classname = Pipeline._get_operator_classname(operator_type)

    return '{0} = {1}({2}, dag=dag)\n'.format(
        task_id,
        operator_classname,
        param_string)

  @staticmethod
  def _get_dag_definition(name, schedule_interval):
    dag_definition = 'dag = DAG(dag_id=\'{0}\', schedule_interval=\'{1}\', ' \
                     'default_args=default_args)\n\n'.format(name,
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
      operator_enum = Operator(task_detail_type).name
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

  def _get_operator_param_value(self, param_name, operator_type, param_value):
    if (operator_type == 'bq'):
      if (param_name in ['query', 'bql']):
        bq_query = self._env.get(param_value)
        if bq_query is not None:
          return bq_query.sql
    return param_value

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


  def execute_async(self, output_options=None, context=None, pipeline_params=None):
    """ Initiate the Airflow job and return an AirflowJob.
    Args:
      output_options: a QueryOutput object describing how to execute the query
      context: an optional Context object providing project_id and credentials. If a specific
          project id or credentials are unspecified, the default ones configured at the global
          level are used.
      pipeline_params: a dictionary containing pipeline parameter types and values.
    Returns:
      A Job object that can wait on creating a table or exporting to a file
      If the output is a table, the Job object additionally has run statistics
      and query results
    Raises:
      Exception if query could not be executed.
    """

    # Default behavior is to print the python
    if output_options is None:
      output_options = QueryOutput.table()

    # First, execute the query into a table, using a temporary one if no name is specified
    batch = output_options.priority == 'low'
    append = output_options.table_mode == 'append'
    overwrite = output_options.table_mode == 'overwrite'
    table_name = output_options.table_name
    context = context or google.datalab.Context.default()
    api = _api.Api(context)
    if table_name is not None:
      table_name = _utils.parse_table_name(table_name, api.project_id)

    sql = self._expanded_sql(sampling)

    try:
      query_result = api.jobs_insert_query(sql, table_name=table_name,
                                           append=append, overwrite=overwrite, batch=batch,
                                           use_cache=output_options.use_cache,
                                           allow_large_results=output_options.allow_large_results,
                                           table_definitions=self.data_sources,
                                           query_params=query_params)
    except Exception as e:
      raise e
    if 'jobReference' not in query_result:
      raise Exception('Unexpected response from server')

    job_id = query_result['jobReference']['jobId']
    if not table_name:
      try:
        destination = query_result['configuration']['query']['destinationTable']
        table_name = (destination['projectId'], destination['datasetId'], destination['tableId'])
      except KeyError:
        # The query was in error
        raise Exception(_utils.format_query_errors(query_result['status']['errors']))

    execute_job = _query_job.QueryJob(job_id, table_name, sql, context=context)

    # If all we need is to execute the query to a table, we're done
    if output_options.type == 'table':
      return execute_job
    # Otherwise, build an async Job that waits on the query execution then carries out
    # the specific export operation
    else:
      export_args = export_kwargs = None
      if output_options.type == 'file':
        if output_options.file_path.startswith('gs://'):
          export_func = execute_job.result().extract
          export_args = [output_options.file_path]
          export_kwargs = {
            'format': output_options.file_format,
            'csv_delimiter': output_options.csv_delimiter,
            'csv_header': output_options.csv_header,
            'compress': output_options.compress_file
          }
        else:
          export_func = execute_job.result().to_file
          export_args = [output_options.file_path]
          export_kwargs = {
            'format': output_options.file_format,
            'csv_delimiter': output_options.csv_delimiter,
            'csv_header': output_options.csv_header
          }
      elif output_options.type == 'dataframe':
        export_func = execute_job.result().to_dataframe
        export_args = []
        export_kwargs = {
          'start_row': output_options.dataframe_start_row,
          'max_rows': output_options.dataframe_max_rows
        }

      # Perform the export operation with the specified parameters
      export_func = google.datalab.utils.async_function(export_func)
      return export_func(*export_args, **export_kwargs)

  def execute(self, context=None):
    """ Copy python spec into GCS for Airflow to pick-up.
    Args:
      context: an optional Context object providing project_id and credentials. If a specific
          project id or credentials are unspecified, the default ones configured at the global
          level are used.
    Returns:
      A Job object that can be used to get the query results, or export to a file or dataframe
    Raises:
      Exception if query could not be executed.
    """
    return self.execute_async(output_options, context=context,
                              pipeline_params=pipeline_params).wait()