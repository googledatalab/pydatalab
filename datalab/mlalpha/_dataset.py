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


"""Implements DataSets that serve two purposes:

1. Recommended way to pass data source to ML packages.
2. All DataSets can be sampled into dataframe for analysis/visualization.
"""

import json
import google.cloud.ml as ml
import numpy as np
import pandas as pd
import random

import datalab.bigquery as bq
import datalab.data


class CsvDataSet(object):
  """DataSet based on CSV files and schema."""

  def __init__(self, file_pattern, schema=None, schema_file=None):
    """
    Args:
      file_pattern: A list of CSV files. or a string. Can contain wildcards in
          file names. Can be local or GCS path.
      schema: A BigQuery schema object in the form of 
          [{'name': 'col1', 'type': 'STRING'},
           {'name': 'col2', 'type': 'INTEGER'}]
      schema_file: A JSON serialized schema file. If schema is None, it will try to load from
          schema_file if not None.
    """
    self._schema = None
    if schema is not None:
      self._schema = schema
    elif schema_file is not None:
      with ml.util._file.open_local_or_gcs(schema_file, 'r') as f:
        self._schema = json.load(f)
    
    self._file_pattern = file_pattern
    self._files = []

  @property
  def file_pattern(self):
    return self._file_pattern

  @property
  def files(self):
    if self._files:
      return self._files

    if isinstance(self._file_pattern, basestring):
      file_list = [self._file_pattern]
    else:
      file_list = self._file_pattern
      
    for file in file_list:
      # glob_files() returns unicode strings which doesn't make DataFlow happy. So str().
      self._files += [str(x) for x in ml.util._file.glob_files(file)]
    
    return self._files
      
  @property
  def schema(self):
    return self._schema  

  def sample(self, n):
    """ Samples data into a Pandas DataFrame.
    Args:
      n: number of sampled counts.
    Returns:
      A dataframe containing sampled data.
    Raises:
      Exception if n is larger than number of rows.
    """
    row_total_count = 0
    row_counts = []
    for file in self.files:
      with ml.util._file.open_local_or_gcs(file, 'r') as f:
        num_lines = sum(1 for line in f)
        row_total_count += num_lines
        row_counts.append(num_lines)

    names = None
    dtype = None
    if self._schema:
      _MAPPINGS = {
        'FLOAT': np.float64,
        'INTEGER': np.int64,
        'TIMESTAMP': np.datetime64,
        'BOOLEAN': np.bool,
      }
      names = [x['name'] for x in self._schema]
      dtype = {x['name']: _MAPPINGS.get(x['type'], object) for x in self._schema}
    
    skip_count = row_total_count - n
    # Get all skipped indexes. These will be distributed into each file.
    # Note that random.sample will raise Exception if skip_count is greater than rows count.
    skip_all = sorted(random.sample(xrange(0, row_total_count), skip_count))
    dfs = []
    for file, row_count in zip(self.files, row_counts):
      skip = [x for x in skip_all if x < row_count]
      skip_all = [x - row_count for x in skip_all if x >= row_count]
      with ml.util._file.open_local_or_gcs(file, 'r') as f:
        dfs.append(pd.read_csv(file, skiprows=skip, names=names, dtype=dtype, header=None))
    return pd.concat(dfs, axis=0, ignore_index=True)


class BigQueryDataSet(object):
  """DataSet based on BigQuery table or query."""

  def __init__(self, sql):
    """
    Args:
      sql: Can be one of:
          A table name.
          A SQL query string.
          A SQL Query module defined with '%%sql --name [module_name]'
    """
    query, _ = datalab.data.SqlModule.get_sql_statement_with_environment(sql, {})
    self._sql = query.sql

  @property
  def sql(self):
    return self._sql
  
  def sample(self, n):
    """Samples data into a Pandas DataFrame. Note that it calls BigQuery so it will
       incur cost.
    Args:
      n: number of sampled counts. Note that the number of counts returned is approximated.
    Returns:
      A dataframe containing sampled data.
    Raises:
      Exception if n is larger than number of rows.
    """
    total = bq.Query('select count(*) from (%s)' % self._sql).results()[0].values()[0]
    if n > total:
      raise ValueError('sample larger than population')
    sampling = bq.Sampling.random(n*100.0/float(total))
    sample = bq.Query(self._sql).sample(sampling=sampling)
    df = sample.to_dataframe()
    return df
