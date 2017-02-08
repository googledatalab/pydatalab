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

import google.datalab
import google.datalab.data
import google.datalab.utils

from ._query_output import QueryOutput
from . import _api
from . import _federated_table
from . import _query_job
from . import _sampling
from . import _udf
from . import _utils


class Query(object):
  """Represents a Query object that encapsulates a BigQuery SQL query.

  This object can be used to execute SQL queries and retrieve results.
  """

  def __init__(self, sql, context=None, values=None, udfs=None, data_sources=None,
               subqueries=None):
    """Initializes an instance of a Query object.

    Args:
      sql: the BigQuery SQL query string to execute, or a SqlStatement object. The latter will
          have any variable references replaced before being associated with the Query (i.e.
          once constructed the SQL associated with a Query is static).

          It is possible to have variable references in a query string too provided the variables
          are passed as keyword arguments to this constructor.

      context: an optional Context object providing project_id and credentials. If a specific
          project id or credentials are unspecified, the default ones configured at the global
          level are used.
      values: a dictionary used to expand variables if passed a SqlStatement or a string with
          variable references.
      udfs: list of UDFs referenced in the SQL.
      data_sources: dictionary of federated (external) tables referenced in the SQL.
      subqueries: list of subqueries referenced in the SQL

    Raises:
      Exception if expansion of any variables failed.
      """
    if context is None:
      context = google.datalab.Context.default()
    self._context = context
    self._api = _api.Api(context)
    self._data_sources = data_sources
    self._udfs = udfs
    self._subqueries = subqueries
    self._values = values

    if data_sources is None:
      data_sources = {}

    self._code = None
    self._imports = []
    if self._values is None:
      self._values = {}

    self._sql = google.datalab.data.SqlModule.expand(sql, self._values)

    def _validate_object(obj):
      if not self._values.__contains__(obj):
        raise Exception('Cannot find object %s.' % obj)

    # Validate subqueries and UDFs when adding them to query
    if self._subqueries:
      for subquery in self._subqueries:
        _validate_object(subquery)
    if self._udfs:
      for udf in self._udfs:
        _validate_object(udf)

    # We need to take care not to include the same UDF code twice so we use sets.
    udfs = set(udfs if udfs else [])
    for value in list(self._values.values()):
      if isinstance(value, _udf.UDF):
        udfs.add(value)
    included_udfs = set([])

    self._external_tables = None
    if len(data_sources):
      self._external_tables = {}
      for name, table in list(data_sources.items()):
        if table.schema is None:
          raise Exception('Referenced external table %s has no known schema' % name)
        self._external_tables[name] = table._to_query_json()

  def _expanded_sql(self, sampling=None):
    """Get the expanded SQL of this object, including all subqueries, UDFs, and external datasources

    Returns:
      The expanded SQL string of this object
    """

    udfs = set()
    subqueries = set()
    expanded_sql = ''

    def _recurse_subqueries(query):
      """Recursively scan subqueries and add their pieces to global scope udfs and subqueries
      """
      if query._subqueries:
        subqueries.update(query._subqueries)
      if query._udfs:
        udfs.update(set(query._udfs))
      if query._subqueries:
        for subquery in query._subqueries:
          _recurse_subqueries(self._values[subquery])

    subqueries_sql = udfs_sql = ''
    _recurse_subqueries(self)

    if udfs:
      expanded_sql += '\n'.join([self._values[udf]._expanded_sql() for udf in udfs])
      expanded_sql += '\n'

    if subqueries:
      expanded_sql += 'WITH ' + \
                      ',\n'.join(['%s AS (%s)' % (sq, self._values[sq]._sql) for sq in subqueries])
      expanded_sql += '\n'

    expanded_sql += sampling(self._sql) if sampling else self._sql

    return expanded_sql

  def _repr_sql_(self):
    """Creates a SQL representation of this object.

    Returns:
      The SQL representation to use when embedding this object into other SQL.
    """
    return '(%s)' % self.sql

  def __str__(self):
    """Creates a string representation of this object.

    Returns:
      The string representation of this object (the unmodified SQL).
    """
    return self._sql

  def __repr__(self):
    """Creates a friendly representation of this object.

    Returns:
      The friendly representation of this object (the unmodified SQL).
    """
    return self._sql

  @property
  def sql(self):
    """ Get the SQL for the query. """
    return self._expanded_sql()

  @property
  def scripts(self):
    """ Get the code for any Javascript UDFs used in the query. """
    return self._code

  def execute_dry_run(self):
    """Dry run a query, to check the validity of the query and return some useful statistics.

    Returns:
      A dict with 'cacheHit' and 'totalBytesProcessed' fields.
    Raises:
      An exception if the query was malformed.
    """
    try:
      query_result = self._api.jobs_insert_query(self.sql, self._code, self._imports, dry_run=True,
                                                 table_definitions=self._external_tables)
    except Exception as e:
      raise e
    return query_result['statistics']['query']

  def execute_async(self, output_options=None, sampling=None):
    """ Initiate the query and return a QueryJob.

    Args:
      output_options: a QueryOutput object describing how to execute the query
      sampling: sampling function to use. No sampling is done if None. See bigquery.Sampling
    Returns:
      A Job object that can wait on creating a table or exporting to a file
      If the output is a table, the Job object additionally has run statistics
      and query results
    Raises:
      Exception if query could not be executed.
    """

    # Default behavior is to execute to a table
    if output_options == None:
      output_options = QueryOutput.table()

    # First, execute the query into a table, using a temporary one if no name is specified
    batch = output_options.priority == 'low'
    append = output_options.table_mode == 'append'
    overwrite = output_options.table_mode == 'overwrite'
    table_name = output_options.table_name
    if table_name is not None:
      table_name = _utils.parse_table_name(table_name, self._api.project_id)

    sql = self._expanded_sql(sampling)

    try:
      query_result = self._api.jobs_insert_query(sql, self._code, self._imports,
                                                 table_name=table_name,
                                                 append=append,
                                                 overwrite=overwrite,
                                                 use_cache=output_options.use_cache,
                                                 batch=batch,
                                                 allow_large_results=output_options.allow_large_results,
                                                 table_definitions=self._external_tables)
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

    execute_job = _query_job.QueryJob(job_id, table_name, sql, context=self._context)

    # If all we need is to execute the query to a table, we're done
    if output_options.type == 'table':
      return execute_job
    # Otherwise, build an async Job that waits on the query execution then carries out
    # the specific export operation
    else:
      export_job = export_args = export_kwargs = None
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

  def execute(self, output_options=None, sampling=None):
    """ Initiate the query and return a QueryJob.

    Args:
      output_options: a QueryOutput object describing how to execute the query
      sampling: sampling function to use. No sampling is done if None. See bigquery.Sampling
    Returns:
      A Job object that can be used to get the query results, or export to a file or dataframe
    Raises:
      Exception if query could not be executed.
    """
    return self.execute_async(output_options, sampling=sampling).wait()

  def to_view(self, view_name):
    """ Create a View from this Query.

    Args:
      view_name: the name of the View either as a string or a 3-part tuple
          (projectid, datasetid, name).

    Returns:
      A View for the Query.
    """
    # Do the import here to avoid circular dependencies at top-level.
    from . import _view
    return _view.View(view_name, self._context).create(self.sql)

