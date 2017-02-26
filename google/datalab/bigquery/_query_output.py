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

"""Implements BigQuery output type functionality."""

class QueryOutput(object):

  @staticmethod
  def table(name=None, mode='create', use_cache=True, priority='interactive',
            allow_large_results=False):
    """ Construct a query output object where the result is a table

    Args:
      name: the result table name as a string or TableName; if None (the default), then a
          temporary table will be used.
      table_mode: one of 'create', 'overwrite' or 'append'. If 'create' (the default), the request
          will fail if the table exists.
      use_cache: whether to use past query results or ignore cache. Has no effect if destination is
          specified (default True).
      priority:one of 'batch' or 'interactive' (default). 'interactive' jobs should be scheduled
          to run quickly but are subject to rate limits; 'batch' jobs could be delayed by as much
          as three hours but are not rate-limited.
      allow_large_results: whether to allow large results; i.e. compressed data over 100MB. This is
          slower and requires a name to be specified) (default False).
    """
    output = QueryOutput()
    output._output_type = 'table'
    output._table_name = name
    output._table_mode = mode
    output._use_cache = use_cache
    output._priority = priority
    output._allow_large_results = allow_large_results
    return output

  @staticmethod
  def file(path, format='csv', csv_delimiter=',', csv_header=True, compress=False,
           use_cache=True):
    """ Construct a query output object where the result is either a local file or a GCS path

    Note that there are two jobs that may need to be run sequentially, one to run the query,
    and the second to extract the resulting table. These are wrapped by a single outer Job.

    If the query has already been executed and you would prefer to get a Job just for the
    extract, you can can call extract[_async] on the QueryResultsTable returned by the query

    Args:
      path: the destination path. Can either be a local or GCS URI (starting with gs://)
      format: the format to use for the exported data; one of 'csv', 'json', or 'avro'
          (default 'csv').
      csv_delimiter: for CSV exports, the field delimiter to use (default ',').
      csv_header: for CSV exports, whether to include an initial header line (default True).
      compress: whether to compress the data on export. Compression is not supported for
          AVRO format (default False). Applies only to GCS URIs.
      use_cache: whether to use cached results or not (default True).
    """
    output = QueryOutput()
    output._output_type = 'file'
    output._file_path = path
    output._file_format = format
    output._csv_delimiter = csv_delimiter
    output._csv_header = csv_header
    output._compress_file = compress
    return output

  @staticmethod
  def dataframe(start_row=0, max_rows=None, use_cache=True):
    """ Construct a query output object where the result is a dataframe

    Args:
      start_row: the row of the table at which to start the export (default 0).
      max_rows: an upper limit on the number of rows to export (default None).
      use_cache: whether to use cached results or not (default True).
    """
    output = QueryOutput()
    output._output_type = 'dataframe'
    output._dataframe_start_row = start_row
    output._dataframe_max_rows = max_rows
    output._use_cache = use_cache
    return output

  def __init__(self):
    """ Create a BigQuery output type object. Do not call this directly; use factory methods. """
    self._output_type = None
    self._table_name = None
    self._table_mode = None
    self._use_cache = None
    self._priority = None
    self._allow_large_results = None
    self._file_path = None
    self._file_format = None
    self._csv_delimiter = None
    self._csv_header = None
    self._compress_file = None
    self._dataframe_start_row = None
    self._dataframe_max_rows = None

  @property
  def type(self):
    return self._output_type

  @property
  def table_name(self):
    return self._table_name

  @property
  def table_mode(self):
    return self._table_mode

  @property
  def use_cache(self):
    return self._use_cache

  @property
  def priority(self):
    return self._priority

  @property
  def allow_large_results(self):
    return self._allow_large_results

  @property
  def file_path(self):
    return self._file_path

  @property
  def file_format(self):
    return self._file_format

  @property
  def csv_delimiter(self):
    return self._csv_delimiter

  @property
  def csv_header(self):
    return self._csv_header

  @property
  def compress_file(self):
    return self._compress_file

  @property
  def dataframe_start_row(self):
    return self._dataframe_start_row

  @property
  def dataframe_max_rows(self):
    return self._dataframe_max_rows

