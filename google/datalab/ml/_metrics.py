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


import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import google.datalab.bigquery as bq

from . import _util


class Metrics(object):
  """Represents a Metrics object that computes metrics from raw evaluation results."""

  def __init__(self, input_csv_pattern=None, headers=None, bigquery=None):
    """
    Args:
      input_csv_pattern: Path to Csv file pattern (with no header). Can be local or GCS path.
      headers: Csv headers. Required if input_csv_pattern is not None.
      bigquery: Can be one of:
          A BigQuery query string.
          A Bigquery table string.
          A Query object defined with '%%bq query --name [query_name]'.

    Raises:
      ValueError if input_csv_pattern is provided but both headers and schema_file are None.
      ValueError if but both input_csv_pattern and bigquery are None.
    """

    self._input_csv_files = None
    self._bigquery = None
    if input_csv_pattern:
      self._input_csv_files = _util.glob_files(input_csv_pattern)
      if not headers:
        raise ValueError('csv requires headers.')
      self._headers = headers
    elif bigquery:
      self._bigquery = bigquery
    else:
      raise ValueError('Either input_csv_pattern or bigquery needs to be provided.')

  @staticmethod
  def from_csv(input_csv_pattern, headers=None, schema_file=None):
    """Create a Metrics instance from csv file pattern.

    Args:
      input_csv_pattern: Path to Csv file pattern (with no header). Can be local or GCS path.
      headers: Csv headers.
      schema_file: Path to a JSON file containing BigQuery schema. Used if "headers" is None.

    Returns:
      a Metrics instance.

    Raises:
      ValueError if both headers and schema_file are None.
    """

    if headers is not None:
      names = headers
    elif schema_file is not None:
      with _util.open_local_or_gcs(schema_file, mode='r') as f:
        schema = json.load(f)
      names = [x['name'] for x in schema]
    else:
      raise ValueError('Either headers or schema_file is needed')

    metrics = Metrics(input_csv_pattern=input_csv_pattern, headers=names)
    return metrics

  @staticmethod
  def from_bigquery(sql):
    """Create a Metrics instance from a bigquery query or table.

    Returns:
      a Metrics instance.

    Args:
      sql: A BigQuery table name or a query.
    """

    if isinstance(sql, bq.Query):
      sql = sql._expanded_sql()

    parts = sql.split('.')
    if len(parts) == 1 or len(parts) > 3 or any(' ' in x for x in parts):
      sql = '(' + sql + ')'  # query, not a table name
    else:
      sql = '`' + sql + '`'  # table name

    metrics = Metrics(bigquery=sql)
    return metrics

  def _get_data_from_csv_files(self):
    """Get data from input csv files."""

    all_df = []
    for file_name in self._input_csv_files:
      with _util.open_local_or_gcs(file_name, mode='r') as f:
        all_df.append(pd.read_csv(f, names=self._headers))
    df = pd.concat(all_df, ignore_index=True)
    return df

  def _get_data_from_bigquery(self, queries):
    """Get data from bigquery table or query."""

    all_df = []
    for query in queries:
      all_df.append(query.execute().result().to_dataframe())
    df = pd.concat(all_df, ignore_index=True)
    return df

  def accuracy(self):
    """Get accuracy numbers for each target and overall.

    Returns:
      A DataFrame with two columns: 'class' and 'accuracy'. It also contains the overall
      accuracy with class being '_all'.

    Raises:
      Exception if the CSV headers do not include 'target' or 'predicted', or BigQuery
      does not return 'target' or 'predicted' column.
    """

    if self._input_csv_files:
      df = self._get_data_from_csv_files()
      if 'target' not in df or 'predicted' not in df:
        raise ValueError('Cannot find "target" or "predicted" column')

      labels = sorted(set(df['target']) | set(df['predicted']))
      accuracy_results = []

      for label in labels:
        correct_count = len(df[(df['target'] == df['predicted']) & (df['target'] == label)])
        total_count = len(df[(df['target'] == label)])
        accuracy_results.append({
            'target': label,
            'accuracy': float(correct_count) / total_count if total_count > 0 else 0,
            'count': total_count
        })

      total_correct_count = len(df[(df['target'] == df['predicted'])])
      if len(df) > 0:
        total_accuracy = float(total_correct_count) / len(df)
        accuracy_results.append({'target': '_all', 'accuracy': total_accuracy, 'count': len(df)})
      return pd.DataFrame(accuracy_results)
    elif self._bigquery:
      query = bq.Query("""
SELECT
  target,
  SUM(CASE WHEN target=predicted THEN 1 ELSE 0 END)/COUNT(*) as accuracy,
  COUNT(*) as count
FROM
  %s
GROUP BY
  target""" % self._bigquery)
      query_all = bq.Query("""
SELECT
  "_all" as target,
  SUM(CASE WHEN target=predicted THEN 1 ELSE 0 END)/COUNT(*) as accuracy,
  COUNT(*) as count
FROM
  %s""" % self._bigquery)

      df = self._get_data_from_bigquery([query, query_all])
      return df

  def roc(self, num_thresholds, target_class, probability_column=None):
    """Get true positive rate, false positive rate values from evaluation results.

    Args:
      num_thresholds: an integer. Number of thresholds.
      target_class: a string indciating the target class, i.e. "daisy" in flower classification.
      probability_column: the name of the probability column. If None, defaults to
          value of target_class.

    Returns:
      A DataFrame with columns: 'tpr', 'fpr', 'threshold' with number of rows
      equal to num_thresholds.

    Raises:
      Exception if the CSV headers do not include 'target' or probability_column, or BigQuery
      does not return 'target' or probability_column column.
    """

    if not probability_column:
      probability_column = target_class

    thresholds = np.linspace(0, 1, num_thresholds + 1)
    if self._input_csv_files:
      df = self._get_data_from_csv_files()
      if 'target' not in df or probability_column not in df:
        raise ValueError('Cannot find "target" or "%s" column' % probability_column)

      total_positive = sum(1 for x in df['target'] if x == target_class)
      total_negative = len(df) - total_positive
      true_positives, false_positives = [], []
      for threshold in thresholds:
        true_positive_count = len(df[(df[probability_column] > threshold) &
                                  (df['target'] == target_class)])
        false_positive_count = len(df[(df[probability_column] > threshold) &
                                   (df['target'] != target_class)])
        true_positives.append(true_positive_count)
        false_positives.append(false_positive_count)

      data = []
      for tp, fp, t in zip(true_positives, false_positives, thresholds):
        tpr = (float)(tp) / total_positive if total_positive > 0. else 0.
        fpr = (float)(fp) / total_negative if total_negative > 0. else 0.
        data.append({'tpr': tpr, 'fpr': fpr, 'threshold': t})
      return pd.DataFrame(data)

    elif self._bigquery:
      true_positive_query = bq.Query("""
        SELECT
          COUNT(*) as true_positive
        FROM
          %s
        CROSS JOIN
          (SELECT * FROM UNNEST ([%s]) as t)
        WHERE
          %s > t AND target = '%s'
        GROUP BY t
        ORDER BY t
      """ % (self._bigquery, ','.join(map(str, thresholds)), probability_column, target_class))

      false_positive_query = bq.Query("""
        SELECT
          COUNT(*) as false_positive
        FROM
          %s
        CROSS JOIN
          (SELECT * FROM UNNEST ([%s]) as t)
        WHERE
          %s > t AND target != '%s'
        GROUP BY t
        ORDER BY t
      """ % (self._bigquery, ','.join(map(str, thresholds)), probability_column, target_class))

      total_positive_query = bq.Query("""
        SELECT
          COUNT(*) as total_positive
        FROM
          %s
        WHERE
          target = '%s'
      """ % (self._bigquery, target_class))

      total_negative_query = bq.Query("""
        SELECT
          COUNT(*) as total_negative
        FROM
          %s
        WHERE
          target != '%s'
      """ % (self._bigquery, target_class))

      true_positives = true_positive_query.execute().result()
      false_positives = false_positive_query.execute().result()
      total_positive = total_positive_query.execute().result()[0]['total_positive']
      total_negative = total_negative_query.execute().result()[0]['total_negative']
      data = []
      for tp, fp, t in zip(true_positives, false_positives, thresholds):
        tpr = (float)(tp['true_positive']) / total_positive if total_positive > 0. else 0.
        fpr = (float)(fp['false_positive']) / total_negative if total_negative > 0. else 0.
        data.append({'tpr': tpr, 'fpr': fpr, 'threshold': t})
      data.append({'tpr': 0., 'fpr': 0., 'threshold': 1.0})
      return pd.DataFrame(data)

  def precision_recall(self, num_thresholds, target_class, probability_column=None):
    """Get precision, recall values from evaluation results.

    Args:
      num_thresholds: an integer. Number of thresholds.
      target_class: a string indciating the target class, i.e. "daisy" in flower classification.
      probability_column: the name of the probability column. If None, defaults to
          value of target_class.

    Returns:
      A DataFrame with columns: 'threshold', 'precision', 'recall' with number of rows
      equal to num_thresholds.

    Raises:
      Exception if the CSV headers do not include 'target' or probability_column, or BigQuery
      does not return 'target' or probability_column column.
    """

    if not probability_column:
      probability_column = target_class

    # threshold = 1.0 is excluded.
    thresholds = np.linspace(0, 1, num_thresholds + 1)[0:-1]
    if self._input_csv_files:
      df = self._get_data_from_csv_files()
      if 'target' not in df or probability_column not in df:
        raise ValueError('Cannot find "target" or "%s" column' % probability_column)

      total_target = sum(1 for x in df['target'] if x == target_class)
      total_predicted = []
      correct_predicted = []
      for threshold in thresholds:
        predicted_count = sum(1 for x in df[probability_column] if x > threshold)
        total_predicted.append(predicted_count)
        correct_count = len(df[(df[probability_column] > threshold) &
                            (df['target'] == target_class)])
        correct_predicted.append(correct_count)

      data = []
      for p, c, t in zip(total_predicted, correct_predicted, thresholds):
        precision = (float)(c) / p if p > 0. else 0.
        recall = (float)(c) / total_target if total_target > 0. else 0.
        data.append({'precision': precision, 'recall': recall, 'threshold': t})
      return pd.DataFrame(data)

    elif self._bigquery:
      total_predicted_query = bq.Query("""
        SELECT
          COUNT(*) as total_predicted
        FROM
          %s
        CROSS JOIN
          (SELECT * FROM UNNEST ([%s]) as t)
        WHERE
          %s > t
        GROUP BY t
        ORDER BY t
      """ % (self._bigquery, ','.join(map(str, thresholds)), probability_column))

      correct_predicted_query = bq.Query("""
        SELECT
          COUNT(*) as correct_predicted
        FROM
          %s
        CROSS JOIN
          (SELECT * FROM UNNEST ([%s]) as t)
        WHERE
          %s > t AND target='%s'
        GROUP BY t
        ORDER BY t
      """ % (self._bigquery, ','.join(map(str, thresholds)), probability_column, target_class))

      total_target_query = bq.Query("""
        SELECT
          COUNT(*) as total_target
        FROM
          %s
        WHERE
          target='%s'
      """ % (self._bigquery, target_class))

      total_predicted = total_predicted_query.execute().result()
      correct_predicted = correct_predicted_query.execute().result()
      total_target = total_target_query.execute().result()[0]['total_target']
      data = []
      for p, c, t in zip(total_predicted, correct_predicted, thresholds):
        precision = ((float)(c['correct_predicted']) / p['total_predicted']
                     if p['total_predicted'] > 0. else 0.)
        recall = (float)(c['correct_predicted']) / total_target if total_target > 0. else 0.
        data.append({'precision': precision, 'recall': recall, 'threshold': t})
      return pd.DataFrame(data)

  def rmse(self):
    """Get RMSE for regression model evaluation results.

    Returns:
      the RMSE float number.

    Raises:
      Exception if the CSV headers do not include 'target' or 'predicted', or BigQuery
      does not return 'target' or 'predicted' column, or if target or predicted is not
      number.
    """

    if self._input_csv_files:
      df = self._get_data_from_csv_files()
      if 'target' not in df or 'predicted' not in df:
        raise ValueError('Cannot find "target" or "predicted" column')

      df = df[['target', 'predicted']].apply(pd.to_numeric)
      # if df is empty or contains non-numeric, scikit learn will raise error.
      mse = mean_squared_error(df['target'], df['predicted'])
      return math.sqrt(mse)
    elif self._bigquery:
      query = bq.Query("""
        SELECT
          SQRT(SUM(ABS(predicted-target) * ABS(predicted-target)) / COUNT(*)) as rmse
        FROM
          %s""" % self._bigquery)
      df = self._get_data_from_bigquery([query])
      if df.empty:
        return None
      return df['rmse'][0]

  def mae(self):
    """Get MAE (Mean Absolute Error) for regression model evaluation results.

    Returns:
      the MAE float number.

    Raises:
      Exception if the CSV headers do not include 'target' or 'predicted', or BigQuery
      does not return 'target' or 'predicted' column, or if target or predicted is not
      number.
    """

    if self._input_csv_files:
      df = self._get_data_from_csv_files()
      if 'target' not in df or 'predicted' not in df:
        raise ValueError('Cannot find "target" or "predicted" column')

      df = df[['target', 'predicted']].apply(pd.to_numeric)
      mae = mean_absolute_error(df['target'], df['predicted'])
      return mae
    elif self._bigquery:
      query = bq.Query("""
        SELECT
          SUM(ABS(predicted-target)) / COUNT(*) as mae
        FROM
          %s""" % self._bigquery)
      df = self._get_data_from_bigquery([query])
      if df.empty:
        return None
      return df['mae'][0]

  def percentile_nearest(self, percentile):
    """Get nearest percentile from regression model evaluation results.

    Args:
      percentile: a 0~100 float number.

    Returns:
      the percentile float number.

    Raises:
      Exception if the CSV headers do not include 'target' or 'predicted', or BigQuery
      does not return 'target' or 'predicted' column, or if target or predicted is not
      number.
    """

    if self._input_csv_files:
      df = self._get_data_from_csv_files()
      if 'target' not in df or 'predicted' not in df:
        raise ValueError('Cannot find "target" or "predicted" column')

      df = df[['target', 'predicted']].apply(pd.to_numeric)
      abs_errors = np.array((df['target'] - df['predicted']).apply(abs))
      return np.percentile(abs_errors, percentile, interpolation='nearest')
    elif self._bigquery:
      query = bq.Query("""
        SELECT
          PERCENTILE_DISC(ABS(predicted-target), %f) OVER() AS percentile
        FROM
          %s
        LIMIT 1""" % (float(percentile) / 100, self._bigquery))
      df = self._get_data_from_bigquery([query])
      if df.empty:
        return None
      return df['percentile'][0]
