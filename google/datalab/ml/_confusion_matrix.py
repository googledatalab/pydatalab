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
import itertools
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


import google.datalab.bigquery as bq

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
          A Bigquery table string.
          A Query object defined with '%%bq query --name [query_name]'.
      The query results or table must include "target", "predicted" columns.
    Returns:
      A ConfusionMatrix that can be plotted.
    Raises:
      ValueError if query results or table does not include 'target' or 'predicted' columns.
    """
    if isinstance(sql, bq.Query):
      sql = sql._expanded_sql()

    parts = sql.split('.')
    if len(parts) == 1 or len(parts) > 3 or any(' ' in x for x in parts):
      sql = '(' + sql + ')'  # query, not a table name
    else:
      sql = '`' + sql + '`'  # table name

    query = bq.Query(
        'SELECT target, predicted, count(*) as count FROM %s group by target, predicted' % sql)
    df = query.execute().result().to_dataframe()
    labels = sorted(set(df['target']) | set(df['predicted']))
    labels_count = len(labels)
    df['target'] = [labels.index(x) for x in df['target']]
    df['predicted'] = [labels.index(x) for x in df['predicted']]
    cm = [[0] * labels_count for i in range(labels_count)]
    for index, row in df.iterrows():
      cm[row['target']][row['predicted']] = row['count']
    return ConfusionMatrix(cm, labels)

  def to_dataframe(self):
    """Convert the confusion matrix to a dataframe.

    Returns:
      A DataFrame with "target", "predicted", "count" columns.
    """

    data = []
    for target_index, target_row in enumerate(self._cm):
      for predicted_index, count in enumerate(target_row):
        data.append((self._labels[target_index], self._labels[predicted_index], count))

    return pd.DataFrame(data, columns=['target', 'predicted', 'count'])

  def plot(self, figsize=None, rotation=45):
    """Plot the confusion matrix.

    Args:
      figsize: tuple (x, y) of ints. Sets the size of the figure
      rotation: the rotation angle of the labels on the x-axis.
    """

    fig, ax = plt.subplots(figsize=figsize)

    plt.imshow(self._cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(self._labels))
    plt.xticks(tick_marks, self._labels, rotation=rotation)
    plt.yticks(tick_marks, self._labels)
    if isinstance(self._cm, list):
      # If cm is created from BigQuery then it is a list.
      thresh = max(max(self._cm)) / 2.
      for i, j in itertools.product(range(len(self._labels)), range(len(self._labels))):
        plt.text(j, i, self._cm[i][j], horizontalalignment="center",
                 color="white" if self._cm[i][j] > thresh else "black")
    else:
      # If cm is created from csv then it is a sklearn's confusion_matrix.
      thresh = self._cm.max() / 2.
      for i, j in itertools.product(range(len(self._labels)), range(len(self._labels))):
        plt.text(j, i, self._cm[i, j], horizontalalignment="center",
                 color="white" if self._cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
