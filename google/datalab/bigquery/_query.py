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

import google.datalab.context
import google.datalab.data
import google.datalab.utils

from . import _query_output
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

  @staticmethod
  def sampling_query(sql, context, fields=None, count=5, sampling=None, udfs=None,
                     data_sources=None):
    """Returns a sampling Query for the SQL object.

    Args:
      sql: the SQL statement (string) or Query object to sample.
      context: a Context object providing project_id and credentials.
      fields: an optional list of field names to retrieve.
      count: an optional count of rows to retrieve which is used if a specific
          sampling is not specified.
      sampling: an optional sampling strategy to apply to the table.
      udfs: array of UDFs referenced in the SQL.
      data_sources: dictionary of federated (external) tables referenced in the SQL.
    Returns:
      A Query object for sampling the table.
    """
    return Query(_sampling.Sampling.sampling_query(sql, fields, count, sampling), context=context,
                 udfs=udfs, data_sources=data_sources)

  def __init__(self, sql, context=None, values=None, udfs=None, data_sources=None, **kwargs):
    """Initializes an instance of a Query object.
       Note that either values or kwargs may be used, but not both.

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
      udfs: array of UDFs referenced in the SQL.
      data_sources: dictionary of federated (external) tables referenced in the SQL.
      kwargs: arguments to use when expanding the variables if passed a SqlStatement
          or a string with variable references.

    Raises:
      Exception if expansion of any variables failed.
      """
    if context is None:
      context = google.datalab.context.Context.default()
    self._context = context
    self._api = _api.Api(context)
    self._data_sources = data_sources
    self._udfs = udfs

    if data_sources is None:
      data_sources = {}

    self._code = None
    self._imports = []
    if values is None:
      values = kwargs

    self._sql = google.datalab.data.SqlModule.expand(sql, values)

    # We need to take care not to include the same UDF code twice so we use sets.
    udfs = set(udfs if udfs else [])
    for value in list(values.values()):
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

  def _repr_sql_(self):
    """Creates a SQL representation of this object.

    Returns:
      The SQL representation to use when embedding this object into other SQL.
    """
    return '(%s)' % self._sql

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
    return self._sql

  @property
  def scripts(self):
    """ Get the code for any Javascript UDFs used in the query. """
    return self._code

  def extract(self, storage_uris, format='csv', csv_delimiter=',', csv_header=True,
              compress=False, use_cache=True, dialect=None, billing_tier=None):
    """Exports the query results to GCS.

    Args:
      storage_uris: the destination URI(s). Can be a single URI or a list.
      format: the format to use for the exported data; one of 'csv', 'json', or 'avro'
          (default 'csv').
      csv_delimiter: for csv exports, the field delimiter to use (default ',').
      csv_header: for csv exports, whether to include an initial header line (default True).
      compress: whether to compress the data on export. Compression is not supported for
          AVRO format (default False).
      use_cache: whether to use cached results or not (default True).
      dialect : {'legacy', 'standard'}, default 'legacy'
          'legacy' : Use BigQuery's legacy SQL dialect.
          'standard' : Use BigQuery's standard SQL (beta), which is
          compliant with the SQL 2011 standard.
      billing_tier: Limits the billing tier for this job. Queries that have resource
          usage beyond this tier will fail (without incurring a charge). If unspecified, this
          will be set to your project default. This can also be used to override your
          project-wide default billing tier on a per-query basis.
    Returns:
      A Job object for the export Job if it was completed successfully; else None.
    Raises:
      An Exception if the query or extract failed.
    """
    return self.execute(use_cache=use_cache, dialect=dialect, billing_tier=billing_tier) \
               .results \
               .extract(storage_uris, format=format, csv_delimiter=csv_delimiter,
                        csv_header=csv_header, compress=compress)

  @google.datalab.utils.async_method
  def extract_async(self, storage_uris, format='csv', csv_delimiter=',', csv_header=True,
                    compress=False, use_cache=True, dialect=None, billing_tier=None):
    """Exports the query results to GCS. Returns a Job immediately.

    Note that there are two jobs that may need to be run sequentially, one to run the query,
    and the second to extract the resulting table. These are wrapped by a single outer Job.

    If the query has already been executed and you would prefer to get a Job just for the
    extract, you can can call extract_async on the QueryResultsTable instead; i.e.:

        query.execute().results.extract_async(...)

    Args:
      storage_uris: the destination URI(s). Can be a single URI or a list.
      format: the format to use for the exported data; one of 'csv', 'json', or 'avro'
          (default 'csv').
      csv_delimiter: for CSV exports, the field delimiter to use (default ',').
      csv_header: for CSV exports, whether to include an initial header line (default True).
      compress: whether to compress the data on export. Compression is not supported for
          AVRO format (default False).
      use_cache: whether to use cached results or not (default True).
      dialect : {'legacy', 'standard'}, default 'legacy'
          'legacy' : Use BigQuery's legacy SQL dialect.
          'standard' : Use BigQuery's standard SQL (beta), which is
          compliant with the SQL 2011 standard.
      billing_tier: Limits the billing tier for this job. Queries that have resource
          usage beyond this tier will fail (without incurring a charge). If unspecified, this
          will be set to your project default. This can also be used to override your
          project-wide default billing tier on a per-query basis.
    Returns:
      A Job for the combined (execute, extract) task that will in turn return the Job object for
      the completed extract task when done; else None.
    Raises:
      An Exception if the query failed.
    """
    return self.extract(storage_uris, format=format, csv_delimiter=csv_delimiter,
                        csv_header=csv_header, use_cache=use_cache, compress=compress,
                        dialect=dialect, billing_tier=billing_tier)

  def to_dataframe(self, start_row=0, max_rows=None, use_cache=True, dialect=None,
                   billing_tier=None):
    """ Exports the query results to a Pandas dataframe.

    Args:
      start_row: the row of the table at which to start the export (default 0).
      max_rows: an upper limit on the number of rows to export (default None).
      use_cache: whether to use cached results or not (default True).
      dialect : {'legacy', 'standard'}, default 'legacy'
          'legacy' : Use BigQuery's legacy SQL dialect.
          'standard' : Use BigQuery's standard SQL (beta), which is
          compliant with the SQL 2011 standard.
      billing_tier: Limits the billing tier for this job. Queries that have resource
          usage beyond this tier will fail (without incurring a charge). If unspecified, this
          will be set to your project default. This can also be used to override your
          project-wide default billing tier on a per-query basis.
    Returns:
      A Pandas dataframe containing the table data.
    """
    return self.execute(use_cache=use_cache, dialect=dialect, billing_tier=billing_tier) \
               .results \
               .to_dataframe(start_row=start_row, max_rows=max_rows)

  def to_file(self, path, format='csv', csv_delimiter=',', csv_header=True, use_cache=True,
              dialect=None, billing_tier=None):
    """Save the results to a local file in CSV format.

    Args:
      path: path on the local filesystem for the saved results.
      format: the format to use for the exported data; currently only 'csv' is supported.
      csv_delimiter: for CSV exports, the field delimiter to use. Defaults to ','
      csv_header: for CSV exports, whether to include an initial header line. Default true.
      use_cache: whether to use cached results or not.
      dialect : {'legacy', 'standard'}, default 'legacy'
          'legacy' : Use BigQuery's legacy SQL dialect.
          'standard' : Use BigQuery's standard SQL (beta), which is
          compliant with the SQL 2011 standard.
      billing_tier: Limits the billing tier for this job. Queries that have resource
          usage beyond this tier will fail (without incurring a charge). If unspecified, this
          will be set to your project default. This can also be used to override your
          project-wide default billing tier on a per-query basis.
    Returns:
      The path to the local file.
    Raises:
      An Exception if the operation failed.
    """
    self.execute(use_cache=use_cache, dialect=dialect, billing_tier=billing_tier) \
        .results \
        .to_file(path, format=format, csv_delimiter=csv_delimiter, csv_header=csv_header)
    return path

  @google.datalab.utils.async_method
  def to_file_async(self, path, format='csv', csv_delimiter=',', csv_header=True, use_cache=True,
                    dialect=None, billing_tier=None):
    """Save the results to a local file in CSV format. Returns a Job immediately.

    Args:
      path: path on the local filesystem for the saved results.
      format: the format to use for the exported data; currently only 'csv' is supported.
      csv_delimiter: for CSV exports, the field delimiter to use. Defaults to ','
      csv_header: for CSV exports, whether to include an initial header line. Default true.
      use_cache: whether to use cached results or not.
      dialect : {'legacy', 'standard'}, default 'legacy'
          'legacy' : Use BigQuery's legacy SQL dialect.
          'standard' : Use BigQuery's standard SQL (beta), which is
          compliant with the SQL 2011 standard.
      billing_tier: Limits the billing tier for this job. Queries that have resource
          usage beyond this tier will fail (without incurring a charge). If unspecified, this
          will be set to your project default. This can also be used to override your
          project-wide default billing tier on a per-query basis.
    Returns:
      A Job for the save that returns the path to the local file on completion.
    Raises:
      An Exception if the operation failed.
    """
    return self.to_file(path, format=format, csv_delimiter=csv_delimiter, csv_header=csv_header,
                        use_cache=use_cache, dialect=dialect, billing_tier=billing_tier)

  def sample(self, count=5, fields=None, sampling=None, use_cache=True, dialect=None,
             billing_tier=None):
    """Retrieves a sampling of rows for the query.

    Args:
      count: an optional count of rows to retrieve which is used if a specific
          sampling is not specified (default 5).
      fields: the list of fields to sample (default None implies all).
      sampling: an optional sampling strategy to apply to the table.
      use_cache: whether to use cached results or not (default True).
      dialect : {'legacy', 'standard'}, default 'legacy'
          'legacy' : Use BigQuery's legacy SQL dialect.
          'standard' : Use BigQuery's standard SQL (beta), which is
          compliant with the SQL 2011 standard.
      billing_tier: Limits the billing tier for this job. Queries that have resource
          usage beyond this tier will fail (without incurring a charge). If unspecified, this
          will be set to your project default. This can also be used to override your
          project-wide default billing tier on a per-query basis.
    Returns:
      A QueryResultsTable containing a sampling of the result set.
    Raises:
      Exception if the query could not be executed or query response was malformed.
    """
    return Query.sampling_query(self._sql, self._context, count=count, fields=fields,
                                sampling=sampling, udfs=self._udfs,
                                data_sources=self._data_sources) \
                .execute(use_cache=use_cache,
                         dialect=dialect,
                         billing_tier=billing_tier) \
                .results

  def execute_dry_run(self, dialect=None, billing_tier=None):
    """Dry run a query, to check the validity of the query and return some useful statistics.

    Args:
      dialect : {'legacy', 'standard'}, default 'legacy'
          'legacy' : Use BigQuery's legacy SQL dialect.
          'standard' : Use BigQuery's standard SQL (beta), which is
          compliant with the SQL 2011 standard.
      billing_tier: Limits the billing tier for this job. Queries that have resource
          usage beyond this tier will fail (without incurring a charge). If unspecified, this
          will be set to your project default. This can also be used to override your
          project-wide default billing tier on a per-query basis.
    Returns:
      A dict with 'cacheHit' and 'totalBytesProcessed' fields.
    Raises:
      An exception if the query was malformed.
    """
    try:
      query_result = self._api.jobs_insert_query(self._sql, self._code, self._imports, dry_run=True,
                                                 table_definitions=self._external_tables, dialect=dialect,
                                                 billing_tier=billing_tier)
    except Exception as e:
      raise e
    return query_result['statistics']['query']

  def execute_async(self, output_options=None, dialect=None, billing_tier=None):
    """ Initiate the query and return a QueryJob.

    Args:
      output_options: a QueryOutput object describing how to execute the query
      dialect : {'legacy', 'standard'}, default 'legacy'
          'legacy' : Use BigQuery's legacy SQL dialect.
          'standard' : Use BigQuery's standard SQL (beta), which is
          compliant with the SQL 2011 standard.
      billing_tier: Limits the billing tier for this job. Queries that have resource
          usage beyond this tier will fail (without incurring a charge). If unspecified, this
          will be set to your project default. This can also be used to override your
          project-wide default billing tier on a per-query basis.
    Returns:
      A Job object that can wait on creating a table or exporting to a file
      If the output is a table, the Job object additionally has run statistics.
    Raises:
      Exception if query could not be executed.
    """

    # Default behavior is to execute to a table
    if output_options == None:
      output_options = QueryOutput.table()

    # First, execute the query into a table, using a temporary one if no name is specified
    batch = output_options.priority == 'low'
    append = output_options.mode == 'append'
    overwrite = output_options.mode == 'overwrite'
    table_name = output_options.name
    if table_name is not None:
      table_name = _utils.parse_table_name(table_name, self._api.project_id)

    try:
      query_result = self._api.jobs_insert_query(self._sql, self._code, self._imports,
                                                 table_name=table_name,
                                                 append=append,
                                                 overwrite=overwrite,
                                                 use_cache=output_options.use_cache,
                                                 batch=batch,
                                                 allow_large_results=output_options.allow_large_results,
                                                 table_definitions=self._external_tables,
                                                 dialect=dialect,
                                                 billing_tier=billing_tier)
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

    execute_job = _query_job.QueryJob(job_id, table_name, self._sql, context=self._context)

    # If all we need is to execute the query to a table, we're done
    if output_options.type == 'table':
      return execute_job
    # Otherwise, build an async Job that waits on the query execution then carries out
    # the specific export operation
    else:
      export_job = export_args = export_kwargs = None
      if output_options.type == 'file':
        if output_options.path.startswith('gs://'):
          export_func = execute_job.results.extract_async
          export_args = [output_options.path]
          export_kwargs = {
                            format: output_options.file_format,
                            csv_delimiter: output_options.csv_delimiter,
                            csv_header: output_options.csv_header,
                            compress: output_options.compress
                          }
        else:
          export_func = execute_job.results.to_file_async
          export_args = [output_options.path]
          export_kwargs = {
                            format: output_options.file_format,
                            csv_delimiter: output_options.csv_delimiter.
                            csv_header: output_options.csv_header
                          }
      elif output_options.type == 'dataframe':
        export_func = execute_job.results.to_dataframe
        export_args = []
        export_kwargs = {
                          start_row: output_options.start_row,
                          max_rows: output_options.max_rows,
                          use_cache: output_options.use_cache
                        }

      # Perform the export operation with the specified parameters
      return google.datalab.utils.async(export_args, export_kwargs)(export_func)

  def execute(self, table_name=None, table_mode='create', use_cache=True, priority='interactive',
              allow_large_results=False, dialect=None, billing_tier=None):
    """ Initiate the query and return a QueryJob.

    Args:
      output_options: a QueryOutput object describing how to execute the query
      dialect : {'legacy', 'standard'}, default 'legacy'
          'legacy' : Use BigQuery's legacy SQL dialect.
          'standard' : Use BigQuery's standard SQL (beta), which is
          compliant with the SQL 2011 standard.
      billing_tier: Limits the billing tier for this job. Queries that have resource
          usage beyond this tier will fail (without incurring a charge). If unspecified, this
          will be set to your project default. This can also be used to override your
          project-wide default billing tier on a per-query basis.
    Returns:
      A Job object that can wait on creating a table or exporting to a file
    Raises:
      Exception if query could not be executed.
    """
    return self.execute_async(table_name=table_name, table_mode=table_mode, use_cache=use_cache,
                             priority=priority, allow_large_results=allow_large_results,
                             dialect=dialect, billing_tier=billing_tier) \
                             .wait()

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
    return _view.View(view_name, self._context).create(self._sql)

