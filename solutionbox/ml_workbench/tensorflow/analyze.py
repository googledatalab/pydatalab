# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import json
import os
import sys
import six
import textwrap
from tensorflow.python.lib.io import file_io

from trainer import feature_transforms as constant
from trainer import feature_analysis as feature_analysis


def parse_arguments(argv):
  """Parse command line arguments.

  Args:
    argv: list of command line arguments, including program name.

  Returns:
    An argparse Namespace object.

  Raises:
    ValueError: for bad parameters
  """
  parser = argparse.ArgumentParser(
      formatter_class=argparse.RawDescriptionHelpFormatter,
      description=textwrap.dedent("""\
          Runs analysis on structured data and produces auxiliary files for
          training. The output files can also be used by the Transform step
          to materialize TF.Examples files, which for some problems can speed up
          training.

          Description of input files
          --------------------------

          1) If using csv files, the --schema parameter must be the file path to
             a schema file. The format of this file must be a valid BigQuery
             schema file, which is a JSON file containing a list of dicts.
             Consider the example schema file below:

             [
                {"name": "column_name_1", "type": "integer"},
                {"name": "column_name_2", "type": "float"},
                {"name": "column_name_3", "type": "string"},
                {"name": "column_name_4", "type": "string"},
             ]

             Note that the column names in the csv file much match the order
             in the schema list. Also, we only support three BigQuery types (
             integer, float, and string).

             If instead of csv files, --bigquery is used, the schema file
             is not needed as this program will extract it from
             the table directly.

          2) --features is a file path to a file describing the
             transformations. Below is an example features file:

             {
                "column_name_1": {"transform": "scale"},
                "column_name_3": {"transform": "target"},
                "column_name_2": {"transform": "one_hot"},
                "new_feature_name": {"transform": "key", "source_column": "column_name_4"},
             }

             The format of the dict is `name`: `transform-dict` where the
             `name` is the name of the transformed feature. The `source_column`
             value lists what column in the input data is the source for this
             transformation. If `source_column` is missing, it is assumed the
             `name` is a source column and the transformed feature will have
             the same name as the input column.

             A list of supported `transform-dict`s is below:

             {"transform": "identity"}: does nothing (for numerical columns).
             {"transform": "scale", "value": x}: scale a numerical column to
                [-a, a]. If value is missing, x defaults to 1.
             {"transform": "one_hot"}: makes a one-hot encoding of a string
                column.
             {"transform": "embedding", "embedding_dim": d}: makes an embedding
                of a string column.
             {"transform": "multi_hot", "separator": ' '}: makes a multi-hot
                encoding of a string column.
             {"transform": "bag_of_words"}: bag of words transform for string
                columns.
             {"transform": "tfidf"}: TFIDF transform for string columns.
             {"transform": "image_to_vec", "checkpoint": "gs://b/o"}: From image
                gs url to embeddings. "checkpoint" is a inception v3 checkpoint.
                If absent, a default checkpoint is used.
             {"transform": "target"}: denotes what column is the target. If the
                schema type of this column is string, a one_hot encoding is
                automatically applied. If type is numerical, a identity transform
                is automatically applied.
             {"transform": "key"}: column contains metadata-like information
                and is not included in the model.

             Note that for tfidf and bag_of_words, the input string is assumed
             to contain text separated by a space. So for example, the string
             "a, b c." has three tokens 'a,', 'b', and 'c.'.
  """))
  parser.add_argument('--cloud',
                      action='store_true',
                      help='Analysis will use cloud services.')
  parser.add_argument('--output',
                      metavar='DIR',
                      type=str,
                      required=True,
                      help='GCS or local folder')

  input_group = parser.add_argument_group(
      title='Data Source Parameters',
      description='schema is only needed if using --csv')

  # CSV input
  input_group.add_argument('--csv',
                           metavar='FILE',
                           type=str,
                           required=False,
                           action='append',
                           help='Input CSV absolute file paths. May contain a '
                                'file pattern.')
  input_group.add_argument('--schema',
                           metavar='FILE',
                           type=str,
                           required=False,
                           help='Schema file path. Only required if using csv files')

  # Bigquery input
  input_group.add_argument('--bigquery',
                           metavar='PROJECT_ID.DATASET.TABLE_NAME',
                           type=str,
                           required=False,
                           help=('Must be in the form project.dataset.table_name'))

  parser.add_argument('--features',
                      metavar='FILE',
                      type=str,
                      required=True,
                      help='Features file path')

  args = parser.parse_args(args=argv[1:])

  if args.cloud:
    if not args.output.startswith('gs://'):
      raise ValueError('--output must point to a location on GCS')
    if (args.csv and
       not all(x.startswith('gs://') for x in args.csv)):
      raise ValueError('--csv must point to a location on GCS')
    if args.schema and not args.schema.startswith('gs://'):
      raise ValueError('--schema must point to a location on GCS')

  if not args.cloud and args.bigquery:
    raise ValueError('--bigquery must be used with --cloud')

  if not ((args.bigquery and args.csv is None and
           args.schema is None) or
          (args.bigquery is None and args.csv and
           args.schema)):
    raise ValueError('either --csv and --schema must both'
                     ' be set or just --bigquery is set')

  return args


def run_cloud_analysis(output_dir, csv_file_pattern, bigquery_table, schema,
                       features):
  """Use BigQuery to analyze input date.

  Only one of csv_file_pattern or bigquery_table should be non-None.

  Args:
    output_dir: output folder
    csv_file_pattern: list of csv file paths, may contain wildcards
    bigquery_table: project_id.dataset_name.table_name
    schema: schema list
    features: features config
  """

  def _execute_sql(sql, table):
    """Runs a BigQuery job and dowloads the results into local memeory.

    Args:
      sql: a SQL string
      table: bq.ExternalDataSource or bq.Table

    Returns:
      A Pandas dataframe.
    """
    import google.datalab.bigquery as bq
    if isinstance(table, bq.ExternalDataSource):
      query = bq.Query(sql, data_sources={'csv_table': table})
    else:
      query = bq.Query(sql)
    return query.execute().result().to_dataframe()

  feature_analysis.expand_defaults(schema, features)  # features are updated.
  inverted_features = feature_analysis.invert_features(features)
  feature_analysis.check_schema_transforms_match(schema, inverted_features)

  import google.datalab.bigquery as bq
  if bigquery_table:
    table_name = '`%s`' % bigquery_table
    table = None
  else:
    table_name = 'csv_table'
    table = bq.ExternalDataSource(
        source=csv_file_pattern,
        schema=bq.Schema(schema))

  # Make a copy of inverted_features and update the target transform to be
  # identity or one hot depending on the schema.
  inverted_features_target = copy.deepcopy(inverted_features)
  for name, transforms in six.iteritems(inverted_features_target):
    transform_set = {x['transform'] for x in transforms}
    if transform_set == set([constant.TARGET_TRANSFORM]):
      target_schema = next(col['type'].lower() for col in schema if col['name'] == name)
      if target_schema in constant.NUMERIC_SCHEMA:
        inverted_features_target[name] = [{'transform': constant.IDENTITY_TRANSFORM}]
      else:
        inverted_features_target[name] = [{'transform': constant.ONE_HOT_TRANSFORM}]

  numerical_vocab_stats = {}
  for col_name, transform_set in six.iteritems(inverted_features_target):
    sys.stdout.write('Analyzing column %s...\n' % col_name)
    sys.stdout.flush()
    # All transforms in transform_set require the same analysis. So look
    # at the first transform.
    transform = next(iter(transform_set))
    if (transform['transform'] in constant.CATEGORICAL_TRANSFORMS or
       transform['transform'] in constant.TEXT_TRANSFORMS):
      if transform['transform'] in constant.TEXT_TRANSFORMS:
        # Split strings on space, then extract labels and how many rows each
        # token is in. This is done by making two temp tables:
        #   SplitTable: each text row is made into an array of strings. The
        #       array may contain repeat tokens
        #   TokenTable: SplitTable with repeated tokens removed per row.
        # Then to flatten the arrays, TokenTable has to be joined with itself.
        # See the sections 'Flattening Arrays' and 'Filtering Arrays' at
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/arrays
        separator = transform.get('separator', ' ')
        sql = ('WITH SplitTable AS '
               '         (SELECT SPLIT({name}, \'{separator}\') as token_array FROM {table}), '
               '     TokenTable AS '
               '         (SELECT ARRAY(SELECT DISTINCT x '
               '                       FROM UNNEST(token_array) AS x) AS unique_tokens_per_row '
               '          FROM SplitTable) '
               'SELECT token, COUNT(token) as token_count '
               'FROM TokenTable '
               'CROSS JOIN UNNEST(TokenTable.unique_tokens_per_row) as token '
               'WHERE LENGTH(token) > 0 '
               'GROUP BY token '
               'ORDER BY token_count DESC, token ASC').format(separator=separator,
                                                              name=col_name,
                                                              table=table_name)
      else:
        # Extract label and frequency
        sql = ('SELECT {name} as token, count(*) as count '
               'FROM {table} '
               'WHERE {name} IS NOT NULL '
               'GROUP BY {name} '
               'ORDER BY count DESC, token ASC').format(name=col_name,
                                                        table=table_name)

      df = _execute_sql(sql, table)

      # Save the vocab
      csv_string = df.to_csv(index=False, header=False)
      file_io.write_string_to_file(
          os.path.join(output_dir, constant.VOCAB_ANALYSIS_FILE % col_name),
          csv_string)
      numerical_vocab_stats[col_name] = {'vocab_size': len(df)}

      # free memeory
      del csv_string
      del df
    elif transform['transform'] in constant.NUMERIC_TRANSFORMS:
      # get min/max/average
      sql = ('SELECT max({name}) as max_value, min({name}) as min_value, '
             'avg({name}) as avg_value from {table}').format(name=col_name,
                                                             table=table_name)
      df = _execute_sql(sql, table)
      numerical_vocab_stats[col_name] = {'min': df.iloc[0]['min_value'],
                                         'max': df.iloc[0]['max_value'],
                                         'mean': df.iloc[0]['avg_value']}
    sys.stdout.write('column %s analyzed.\n' % col_name)
    sys.stdout.flush()

  # get num examples
  sql = 'SELECT count(*) as num_examples from {table}'.format(table=table_name)
  df = _execute_sql(sql, table)
  num_examples = df.iloc[0]['num_examples']

  # Write the stats file.
  stats = {'column_stats': numerical_vocab_stats, 'num_examples': num_examples}
  file_io.write_string_to_file(
      os.path.join(output_dir, constant.STATS_FILE),
      json.dumps(stats, indent=2, separators=(',', ': ')))

  feature_analysis.save_schema_features(schema, features, output_dir)


def main(argv=None):
  args = parse_arguments(sys.argv if argv is None else argv)

  if args.schema:
    schema = json.loads(
        file_io.read_file_to_string(args.schema).decode())
  else:
    import google.datalab.bigquery as bq
    schema = bq.Table(args.bigquery).schema._bq_schema
  features = json.loads(
      file_io.read_file_to_string(args.features).decode())

  file_io.recursive_create_dir(args.output)

  if args.cloud:
    run_cloud_analysis(
        output_dir=args.output,
        csv_file_pattern=args.csv,
        bigquery_table=args.bigquery,
        schema=schema,
        features=features)
  else:
    feature_analysis.run_local_analysis(
        output_dir=args.output,
        csv_file_pattern=args.csv,
        schema=schema,
        features=features)


if __name__ == '__main__':
  main()
