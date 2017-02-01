# Copyright 2016 Google Inc. All rights reserved.
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

"""Implements usefule CSV utilities."""
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import next
from builtins import str as newstr
from builtins import range
from builtins import object


import csv
import os
import pandas as pd
import random

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import tempfile
import datalab.storage
import datalab.utils


_MAX_CSV_BYTES = 10000000


class Csv(object):
  """Represents a CSV file in GCS or locally with same schema.
  """
  def __init__(self, path, delimiter=b','):
    """Initializes an instance of a Csv instance.
    Args:
      path: path of the Csv file.
      delimiter: the separator used to parse a Csv line.
    """
    self._path = path
    self._delimiter = delimiter

  @property
  def path(self):
    return self._path

  @staticmethod

  def _read_gcs_lines(path, max_lines=None):
    return datalab.storage.Item.from_url(path).read_lines(max_lines)

  @staticmethod

  def _read_local_lines(path, max_lines=None):
    lines = []
    for line in open(path):
      if max_lines is not None and len(lines) >= max_lines:
        break
      lines.append(line)
    return lines

  def _is_probably_categorical(self, column):
    if newstr(column.dtype) != 'object':
      # only string types (represented in DataFrame as object) can potentially be categorical
      return False
    if len(max(column, key=lambda p: len(newstr(p)))) > 100:
      return False  # value too long to be a category
    if len(set(column)) > 100:
      return False  # too many unique values to be a category
    return True

  def browse(self, max_lines=None, headers=None):
    """Try reading specified number of lines from the CSV object.
    Args:
      max_lines: max number of lines to read. If None, the whole file is read
      headers: a list of strings as column names. If None, it will use "col0, col1..."
    Returns:
      A pandas DataFrame with the schema inferred from the data.
    Raises:
      Exception if the csv object cannot be read or not enough lines to read, or the
      headers size does not match columns size.
    """
    if self.path.startswith('gs://'):
      lines = Csv._read_gcs_lines(self.path, max_lines)
    else:
      lines = Csv._read_local_lines(self.path, max_lines)
    if len(lines) == 0:
      return pd.DataFrame(columns=headers)
    columns_size = len(next(csv.reader([lines[0]], delimiter=self._delimiter)))
    if headers is None:
      headers = ['col' + newstr(e) for e in range(columns_size)]
    if len(headers) != columns_size:
      raise Exception('Number of columns in CSV do not match number of headers')
    buf = StringIO()
    for line in lines:
      buf.write(line)
      buf.write('\n')
    buf.seek(0)
    df = pd.read_csv(buf, names=headers, delimiter=self._delimiter)
    for key, col in df.iteritems():
      if self._is_probably_categorical(col):
        df[key] = df[key].astype('category')
    return df

  def _create_federated_table(self, skip_header_rows):
    import datalab.bigquery as bq
    df = self.browse(1, None)
    # read each column as STRING because we only want to sample rows.
    schema_train = bq.Schema([{'name': name, 'type': 'STRING'} for name in df.keys()])
    options = bq.CSVOptions(skip_leading_rows=(1 if skip_header_rows == True else 0))
    return bq.FederatedTable.from_storage(self.path,
                                          csv_options=options,
                                          schema=schema_train,
                                          max_bad_records=0)

  def _get_gcs_csv_row_count(self, federated_table):
    import datalab.bigquery as bq
    results = bq.Query('SELECT count(*) from data',
                       data_sources={'data': federated_table}).results()
    return results[0].values()[0]

  def sample_to(self, count, skip_header_rows, strategy, target):
    """Sample rows from GCS or local file and save results to target file.
    Args:
      count: number of rows to sample. If strategy is "BIGQUERY", it is used as approximate number.
      skip_header_rows: whether to skip first row when reading from source.
      strategy: can be "LOCAL" or "BIGQUERY". If local, the sampling happens in local memory,
          and number of resulting rows matches count. If BigQuery, sampling is done
          with BigQuery in cloud, and the number of resulting rows will be approximated to
          count.
      target: The target file path, can be GCS or local path.
    Raises:
      Exception if strategy is "BIGQUERY" but source is not a GCS path.
    """
    # TODO(qimingj) Add unit test
    # Read data from source into DataFrame.
    if strategy == 'BIGQUERY':
      import datalab.bigquery as bq
      if not self.path.startswith('gs://'):
        raise Exception('Cannot use BIGQUERY if data is not in GCS')
      federated_table = self._create_federated_table(skip_header_rows)
      row_count = self._get_gcs_csv_row_count(federated_table)
      query = bq.Query('SELECT * from data', data_sources={'data': federated_table})
      sampling = bq.Sampling.random(count*100/float(row_count))
      sample = query.sample(sampling=sampling)
      df = sample.to_dataframe()
    elif strategy == 'LOCAL':
      local_file = self.path
      if self.path.startswith('gs://'):
        local_file = tempfile.mktemp()
        datalab.utils.gcs_copy_file(self.path, local_file)
      with open(local_file) as f:
        row_count = sum(1 for line in f)
      start_row = 1 if skip_header_rows == True else 0
      skip_count = row_count - count - 1 if skip_header_rows == True else row_count - count
      skip = sorted(random.sample(xrange(start_row, row_count), skip_count))
      header_row = 0 if skip_header_rows == True else None
      df = pd.read_csv(local_file, skiprows=skip, header=header_row, delimiter=self._delimiter)
      if self.path.startswith('gs://'):
        os.remove(local_file)
    else:
      raise Exception('strategy must be BIGQUERY or LOCAL')
    # Write to target.
    if target.startswith('gs://'):
      with tempfile.NamedTemporaryFile() as f:
        df.to_csv(f, header=False, index=False)
        f.flush()
        datalab.utils.gcs_copy_file(f.name, target)
    else:
      with open(target, 'w') as f:
        df.to_csv(f, header=False, index=False, sep=str(self._delimiter))
