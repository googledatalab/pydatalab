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


import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

import datalab.bigquery as bq
import datalab.data as data

from . import _util


class ConfusionMatrix(object):
  """Represents a confusion matrix."""

  def __init__(self, cm, labels):
    """
    Args:
      cm: a 2-dimensional matrix with row index being target, column index being predicted,
          and values being count.
      labels: the labels whose order matches the row/column indexes.
    """
    self._cm = cm
    self._labels = labels

  @staticmethod
  def from_csv(input_csv, headers=None, schema_file=None):
    """Create a ConfusionMatrix from a csv file.
    Args:
      input_csv: Path to a Csv file (with no header). Can be local or GCS path.
          May contain wildcards.
      headers: Csv headers. If present, it must include 'target' and 'predicted'.
      schema_file: Path to a JSON file containing BigQuery schema. Used if "headers" is None.
          If present, it must include 'target' and 'predicted' columns.
    Returns:
      A ConfusionMatrix that can be plotted.
    Raises:
      ValueError if both headers and schema_file are None, or it does not include 'target'
          or 'predicted' columns.
    """

    if headers is not None:
      names = headers
    elif schema_file is not None:
      with _util.open_local_or_gcs(schema_file, mode='r') as f:
        schema = json.load(f)
      names = [x['name'] for x in schema]
    else:
      raise ValueError('Either headers or schema_file is needed')

    all_files = _util.glob_files(input_csv)
    all_df = []
    for file_name in all_files:
      with _util.open_local_or_gcs(file_name, mode='r') as f:
        all_df.append(pd.read_csv(f, names=names))
    df = pd.concat(all_df, ignore_index=True)
    
    if 'target' not in df or 'predicted' not in df:
      raise ValueError('Cannot find "target" or "predicted" column')

    labels = sorted(set(df['target']) | set(df['predicted']))
    cm = confusion_matrix(df['target'], df['predicted'], labels=labels)
    return ConfusionMatrix(cm, labels)

  @staticmethod
  def from_bigquery(sql):
    """Create a ConfusionMatrix from a BigQuery table or query.
    Args:
      sql: Can be one of:
          A SQL query string.
          A SQL Query module defined with '%%sql --name [module_name]'.
          A Bigquery table.
      The query results or table must include "target", "predicted" columns.
    Returns:
      A ConfusionMatrix that can be plotted.
    Raises:
      ValueError if query results or table does not include 'target' or 'predicted' columns.
    """

    query, _ = data.SqlModule.get_sql_statement_with_environment(sql, {})
    sql = ('SELECT target, predicted, count(*) as count FROM (%s) group by target, predicted'
        % query.sql)
    df = bq.Query(sql).results().to_dataframe()
    labels = sorted(set(df['target']) | set(df['predicted']))
    labels_count = len(labels)
    df['target'] = [labels.index(x) for x in df['target']]
    df['predicted'] = [labels.index(x) for x in df['predicted']]
    cm = [[0]*labels_count for i in range(labels_count)]
    for index, row in df.iterrows():
      cm[row['target']][row['predicted']] = row['count']
    return ConfusionMatrix(cm, labels)

  def plot(self):
    """Plot the confusion matrix."""

    plt.imshow(self._cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(self._labels))
    plt.xticks(tick_marks, self._labels, rotation=45)
    plt.yticks(tick_marks, self._labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
  
