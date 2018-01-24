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

"""Implements Query BigQuery API."""
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import object

import datetime
import google.datalab
import google.datalab.data
import google.datalab.utils
import six

from ._query_output import QueryOutput
from . import _api
from . import _query_job
from . import _udf
from . import _utils
from . import _external_data_source


class Query(object):
  """Represents a Query object that encapsulates a BigQuery SQL query.

  This object can be used to execute SQL queries and retrieve results.
  """

  def __init__(self, sql, env=None, udfs=None, data_sources=None, subqueries=None):
    """Initializes an instance of a Query object.

    Args:
      sql: the BigQuery SQL query string to execute
      env: a dictionary containing objects from the query execution context, used to get references
          to UDFs, subqueries, and external data sources referenced by the query
      udfs: list of UDFs names referenced in the SQL, or dictionary of names and UDF objects
      data_sources: list of external data sources names referenced in the SQL, or dictionary of
          names and data source objects
      subqueries: list of subqueries names referenced in the SQL, or dictionary of names and
          Query objects

    Raises:
      Exception if expansion of any variables failed.
      """
    self._sql = sql
    self._udfs = []
    self._subqueries = []
    self._data_sources = []
    self._env = env or {}

    # Validate given list or dictionary of objects that they are of correct type
    # and add them to the target dictionary
    def _expand_objects(obj_container, obj_type, target_list):
      for item in obj_container:
        # for a list of objects, we should find these objects in the given environment
        if isinstance(obj_container, list):
          value = self._env.get(item)
          if value is None:
            raise Exception('Cannot find object %s' % item)

        # for a dictionary of objects, each pair must be a string and object of the expected type
        elif isinstance(obj_container, dict):
          value = obj_container[item]
          if not isinstance(value, obj_type):
            raise Exception('Expected type: %s, found: %s.' % (obj_type, type(value)))

        else:
          raise Exception('Unexpected container for type %s. Expected a list or dictionary'
                          % obj_type)

        target_list.append((item, value))

    if subqueries:
      _expand_objects(subqueries, Query, self._subqueries)
    if udfs:
      _expand_objects(udfs, _udf.UDF, self._udfs)
    if data_sources:
      _expand_objects(data_sources, _external_data_source.ExternalDataSource, self._data_sources)

    if len(self._data_sources) > 1:
      raise Exception('Only one temporary external datasource is supported in queries.')

  @staticmethod
  def from_view(view):
    """ Return a Query for the given View object

    Args:
      view: the View object to construct a Query out of

    Returns:
      A Query object with the same sql as the given View object
    """
    return Query('SELECT * FROM %s' % view._repr_sql_())

  @staticmethod
  def from_table(table, fields=None):
    """ Return a Query for the given Table object

    Args:
      table: the Table object to construct a Query out of
      fields: the fields to return. If None, all fields will be returned. This can be a string
          which will be injected into the Query after SELECT, or a list of field names.

    Returns:
      A Query object that will return the specified fields from the records in the Table.
    """
    if fields is None:
      fields = '*'
    elif isinstance(fields, list):
      fields = ','.join(fields)
    return Query('SELECT %s FROM %s' % (fields, table._repr_sql_()))

  def _expanded_sql(self, sampling=None):
    """Get the expanded SQL of this object, including all subqueries, UDFs, and external datasources

    Returns:
      The expanded SQL string of this object
    """

    # use lists to preserve the order of subqueries, bigquery will not like listing subqueries
    # out of order if they depend on each other. for example. the following will be rejected:
    # WITH q2 as (SELECT * FROM q1),
    #      q1 as (SELECT * FROM mytable),
    # SELECT * FROM q2
    # so when we're getting the dependencies, use recursion into a list to maintain the order
    udfs = []
    subqueries = []
    expanded_sql = ''

    def _recurse_subqueries(query):
      """Recursively scan subqueries and add their pieces to global scope udfs and subqueries
      """
      if query._subqueries:
        for subquery in query._subqueries:
          _recurse_subqueries(subquery[1])
        subqueries.extend([s for s in query._subqueries if s not in subqueries])
      if query._udfs:
        # query._udfs is a list of (name, UDF) tuples; we just want the UDF.
        udfs.extend([u[1] for u in query._udfs if u[1] not in udfs])

    _recurse_subqueries(self)

    if udfs:
      expanded_sql += '\n'.join([udf._expanded_sql() for udf in udfs])
      expanded_sql += '\n'

    def _indent_query(subquery):
      return '  ' + subquery._sql.replace('\n', '\n  ')

    if subqueries:
      expanded_sql += 'WITH ' + \
                      '\n),\n'.join(['%s AS (\n%s' % (sq[0], _indent_query(sq[1]))
                                     for sq in subqueries])
      expanded_sql += '\n)\n\n'

    expanded_sql += sampling(self._sql) if sampling else self._sql

    return expanded_sql

  def _repr_sql_(self):
    """Creates a SQL representation of this object.

    Returns:
      The SQL representation to use when embedding this object into other SQL.
    """
    return '(%s)' % self.sql

  def __repr__(self):
    """Creates a friendly representation of this object.

    Returns:
      The friendly representation of this object (the unmodified SQL).
    """
    return 'BigQuery Query - %s' % self._sql

  @property
  def sql(self):
    """ Get the SQL for the query. """
    return self._expanded_sql()

  @property
  def udfs(self):
    """ Get a dictionary of UDFs referenced by the query."""
    return dict(self._udfs)

  @property
  def subqueries(self):
    """ Get a dictionary of subqueries referenced by the query."""
    return dict(self._subqueries)

  @property
  def data_sources(self):
    """ Get a dictionary of external data sources referenced by the query."""
    return dict(self._data_sources)

  def dry_run(self, context=None, query_params=None):
    """Dry run a query, to check the validity of the query and return some useful statistics.

    Args:
      context: an optional Context object providing project_id and credentials. If a specific
          project id or credentials are unspecified, the default ones configured at the global
          level are used.
      query_params: a dictionary containing query parameter types and values, passed to BigQuery.

    Returns:
      A dict with 'cacheHit' and 'totalBytesProcessed' fields.
    Raises:
      An exception if the query was malformed.
    """

    context = context or google.datalab.Context.default()
    api = _api.Api(context)
    try:
      query_result = api.jobs_insert_query(self.sql, dry_run=True,
                                           table_definitions=self.data_sources,
                                           query_params=query_params)
    except Exception as e:
      raise e
    return query_result['statistics']['query']

  def execute_async(self, output_options=None, sampling=None, context=None, query_params=None):
    """ Initiate the query and return a QueryJob.

    Args:
      output_options: a QueryOutput object describing how to execute the query
      sampling: sampling function to use. No sampling is done if None. See bigquery.Sampling
      context: an optional Context object providing project_id and credentials. If a specific
          project id or credentials are unspecified, the default ones configured at the global
          level are used.
      query_params: a dictionary containing query parameter types and values, passed to BigQuery.
    Returns:
      A Job object that can wait on creating a table or exporting to a file
      If the output is a table, the Job object additionally has run statistics
      and query results
    Raises:
      Exception if query could not be executed.
    """

    # Default behavior is to execute to a table
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

  def execute(self, output_options=None, sampling=None, context=None, query_params=None):
    """ Initiate the query and return a QueryJob.

    Args:
      output_options: a QueryOutput object describing how to execute the query
      sampling: sampling function to use. No sampling is done if None. See bigquery.Sampling
      context: an optional Context object providing project_id and credentials. If a specific
          project id or credentials are unspecified, the default ones configured at the global
          level are used.
    Returns:
      A Job object that can be used to get the query results, or export to a file or dataframe
    Raises:
      Exception if query could not be executed.
    """
    return self.execute_async(output_options, sampling=sampling, context=context,
                              query_params=query_params).wait()

  @staticmethod
  def get_query_parameters(config_parameters, date_time=datetime.datetime.now()):
    """ Merge the given parameters with the airflow macros. Enables macros (like '@_ds') in sql.

    Args:
      config_parameters: The user-specified list of parameters in the cell-body.
      date_time: The timestamp at which the parameters need to be evaluated. E.g. when the table
          is <project-id>.<dataset-id>.logs_%(_ds)s, the '_ds' evaluates to the current date-time.

    Returns:
      A list of query parameters that are in the format for the BQ service
    """
    merged_parameters = Query.merge_parameters(config_parameters, date_time=date_time,
                                               macros=False, types_and_values=True)
    # We're exposing a simpler schema format than the one actually required by BigQuery to make
    # magics easier. We need to convert between the two formats
    parsed_params = []
    for key, value in merged_parameters.items():
      parsed_params.append({
        'name': key,
        'parameterType': {
          'type': value['type']
        },
        'parameterValue': {
          'value': value['value']
        }
      })
    return parsed_params

  @staticmethod
  def resolve_parameters(value, parameters, date_time=datetime.datetime.now(), macros=False):
    """ Resolve a format modifier with the corresponding value.

    Args:
      value: The string (path, table, or any other artifact in a cell_body) which may have format
          modifiers. E.g. a table name could be <project-id>.<dataset-id>.logs_%(_ds)s
      parameters: The user-specified list of parameters in the cell-body.
      date_time: The timestamp at which the parameters need to be evaluated. E.g. when the table
          is <project-id>.<dataset-id>.logs_%(_ds)s, the '_ds' evaluates to the current date-time.
      macros: When true, the format modifers in the value are replaced with the corresponding
          airflow macro equivalents (like '{{ ds }}'. When false, the actual values are used (like
          '2015-12-12'.

    Returns:
      The resolved value, i.e. the value with the format modifiers replaced with the corresponding
          parameter-values. E.g. if value is <project-id>.<dataset-id>.logs_%(_ds)s, the returned
          value is something like <project-id>.<dataset-id>.logs_2017-12-21
    """
    merged_parameters = Query.merge_parameters(parameters, date_time=date_time, macros=macros,
                                               types_and_values=False)
    return Query._resolve_parameters(value, merged_parameters)

  @staticmethod
  def _resolve_parameters(operator_param_value, merged_parameters):
    """ Resolve a format modifier with the corresponding value.

    Args:
      operator_param_value: The object with the format-modifiers that need to be evaluated. This
          could either be a string, or a more complex type like a list or a dict. This function
          will recursively replace the format-modifiers from all the string values that it can
          find.
      merged_parameters: The full set of parameters that include the user-specified list of
          parameters from the cell-body, and the built-in airflow macros (either the macros or the
          evaluated-values).

    Returns:
      The resolved value, i.e. the value with the format modifiers replaced with the corresponding
          parameter-values. E.g. if value is <project-id>.<dataset-id>.logs_%(_ds)s, the returned
          value could be <project-id>.<dataset-id>.logs_2017-12-21.
    """
    if isinstance(operator_param_value, list):
      return [Query._resolve_parameters(item, merged_parameters)
              for item in operator_param_value]
    if isinstance(operator_param_value, dict):
      return {Query._resolve_parameters(k, merged_parameters): Query._resolve_parameters(
        v, merged_parameters) for k, v in operator_param_value.items()}
    if isinstance(operator_param_value, six.string_types) and merged_parameters:
      return operator_param_value % merged_parameters
    return operator_param_value

  @staticmethod
  def _airflow_macro_formats(date_time, macros, types_and_values):
    """ Return a mapping from airflow macro names (prefixed with '_') to values

    Args:
      date_time: The timestamp at which the macro values need to be evaluated. This is only
          applicable when types_and_values = True
      macros: If true, the items in the returned dict are the macro strings (like '_ds': '{{ ds }}')
      types_and_values: If true, the values in the returned dict are dictionaries of the types and
          the values of the parameters (i.e like '_ds': {'type': STRING, 'value': 2017-12-21})
    Returns:
      The resolved value, i.e. the value with the format modifiers replaced with the corresponding
          parameter-values. E.g. if value is <project-id>.<dataset-id>.logs_%(_ds)s, the returned
          value could be <project-id>.<dataset-id>.logs_2017-12-21.
    """
    day = date_time.date()
    airflow_macros = {
      # the datetime formatted as YYYY-MM-DD
      '_ds': {'type': 'STRING', 'value': day.isoformat(), 'macro': '{{ ds }}'},
      # the full ISO-formatted timestamp YYYY-MM-DDTHH:MM:SS.mmmmmm
      '_ts': {'type': 'STRING', 'value': date_time.isoformat(), 'macro': '{{ ts }}'},
      # the datetime formatted as YYYYMMDD (i.e. YYYY-MM-DD with 'no dashes')
      '_ds_nodash': {'type': 'STRING', 'value': day.strftime('%Y%m%d'),
                     'macro': '{{ ds_nodash }}'},
      # the timestamp formatted as YYYYMMDDTHHMMSSmmmmmm (i.e full ISO-formatted timestamp
      # YYYY-MM-DDTHH:MM:SS.mmmmmm with no dashes or colons).
      '_ts_nodash': {'type': 'STRING', 'value': date_time.strftime('%Y%m%d%H%M%S%f'),
                     'macro': '{{ ts_nodash }}'},
      '_ts_year': {'type': 'STRING', 'value': day.strftime('%Y'),
                   'macro': """{{ '{:04d}'.format(execution_date.year) }}"""},
      '_ts_month': {'type': 'STRING', 'value': day.strftime('%m'),
                    'macro': """{{ '{:02d}'.format(execution_date.month) }}"""},
      '_ts_day': {'type': 'STRING', 'value': day.strftime('%d'),
                  'macro': """{{ '{:02d}'.format(execution_date.day) }}"""},
      '_ts_hour': {'type': 'STRING', 'value': date_time.strftime('%H'),
                   'macro': """{{ '{:02d}'.format(execution_date.hour) }}"""},
      '_ts_minute': {'type': 'STRING', 'value': date_time.strftime('%M'),
                     'macro': """{{ '{:02d}'.format(execution_date.minute) }}"""},
      '_ts_second': {'type': 'STRING', 'value': date_time.strftime('%S'),
                     'macro': """{{ '{:02d}'.format(execution_date.second) }}"""},
    }

    if macros:
      return {key: value['macro'] for key, value in airflow_macros.items()}

    if types_and_values:
      return {
        key: {
          'type': item['type'],
          'value': item['value']
        } for key, item in airflow_macros.items()
      }

    # By default only return values
    return {key: value['value'] for key, value in airflow_macros.items()}

  @staticmethod
  def merge_parameters(parameters, date_time, macros, types_and_values):
    """ Merge Return a mapping from airflow macro names (prefixed with '_') to values

    Args:
      date_time: The timestamp at which the macro values need to be evaluated. This is only
          applicable when types_and_values = True
      macros: If true, the values in the returned dict are the macro strings (like '{{ ds }}')

    Returns:
      The resolved value, i.e. the value with the format modifiers replaced with the corresponding
          parameter-values. E.g. if value is <project-id>.<dataset-id>.logs_%(_ds)s, the returned
          value could be <project-id>.<dataset-id>.logs_2017-12-21.
    """
    merged_parameters = Query._airflow_macro_formats(date_time=date_time, macros=macros,
                                                     types_and_values=types_and_values)
    if parameters:
      if types_and_values:
        parameters = {
          item['name']: {'value': item['value'], 'type': item['type']}
          for item in parameters
        }
      else:  # macros = True, or the default (i.e. just values)
        parameters = {item['name']: item['value'] for item in parameters}

      merged_parameters.update(parameters)
    return merged_parameters
