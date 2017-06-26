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
import collections
import copy
import csv
import json
import os
import pandas as pd
import sys
import six
import textwrap
from tensorflow.python.lib.io import file_io

from trainer import feature_transforms as constant


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

          1) If using csv files, the --csv-schema-file must be the file path to
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

             If instead of csv files, --bigquery-table is used, the schema file
             is not needed as this program will extract it from
             the table directly.

          2) --features-file is a file path to a file describing the
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
             {"transform": "bag_of_words"}: bag of words transform for string
                columns.
             {"transform": "tfidf"}: TFIDF transform for string columns.
             {"transform": "image_to_vec"}: From image gs url to embeddings.
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
  parser.add_argument('--output-dir',
                      metavar='FOLDER',
                      type=str,
                      required=True,
                      help='GCS or local folder')

  # CSV inputs
  parser.add_argument('--csv-file-pattern',
                      metavar='FILE',
                      type=str,
                      required=False,
                      action='append',
                      help=('Input CSV file names. May contain a file pattern. '
                            'File prefix must include absolute file path.'))
  parser.add_argument('--csv-schema-file',
                      metavar='FILE',
                      type=str,
                      required=False,
                      help=('BigQuery json schema file path'))

  # If using bigquery table
  parser.add_argument('--bigquery-table',
                      type=str,
                      required=False,
                      help=('Must be in the form project.dataset.table_name'))

  parser.add_argument('--features-file',
                      metavar='FILE',
                      type=str,
                      required=True,
                      help='Features file path')

  args = parser.parse_args(args=argv[1:])

  if args.cloud:
    if not args.output_dir.startswith('gs://'):
      raise ValueError('--output-dir must point to a location on GCS')
    if (args.csv_file_pattern and
       not all(x.startswith('gs://') for x in args.csv_file_pattern)):
      raise ValueError('--csv-file-pattern must point to a location on GCS')
    if args.csv_schema_file and not args.csv_schema_file.startswith('gs://'):
      raise ValueError('--csv-schema-file must point to a location on GCS')

  if not args.cloud and args.bigquery_table:
    raise ValueError('--bigquery-table must be used with --cloud')

  if not ((args.bigquery_table and args.csv_file_pattern is None and
           args.csv_schema_file is None) or
          (args.bigquery_table is None and args.csv_file_pattern and
           args.csv_schema_file)):
    raise ValueError('either --csv-schema-file and --csv-file-pattern must both'
                     ' be set or just --bigquery-table is set')

  return args


def run_cloud_analysis(output_dir, csv_file_pattern, bigquery_table, schema,
                       inverted_features):
  """Use BigQuery to analyze input date.

  Only one of csv_file_pattern or bigquery_table should be non-None.

  Args:
    output_dir: output folder
    csv_file_pattern: list of csv file paths, may contain wildcards
    bigquery_table: project_id.dataset_name.table_name
    schema: schema list
    inverted_features: inverted_features dict
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
  for name, transform_set in six.iteritems(inverted_features_target):
    if transform_set == set([constant.TARGET_TRANSFORM]):
      target_schema = next(col['type'].lower() for col in schema if col['name'] == name)
      if target_schema in constant.NUMERIC_SCHEMA:
        inverted_features_target[name] = {constant.IDENTITY_TRANSFORM}
      else:
        inverted_features_target[name] = {constant.ONE_HOT_TRANSFORM}

  numerical_vocab_stats = {}
  for col_name, transform_set in six.iteritems(inverted_features_target):
    sys.stdout.write('Analyzing column %s...' % col_name)
    sys.stdout.flush()
    # All transforms in transform_set require the same analysis. So look
    # at the first transform.
    transform_name = next(iter(transform_set))
    if (transform_name in constant.CATEGORICAL_TRANSFORMS or
       transform_name in constant.TEXT_TRANSFORMS):
      if transform_name in constant.TEXT_TRANSFORMS:
        # Split strings on space, then extract labels and how many rows each
        # token is in. This is done by making two temp tables:
        #   SplitTable: each text row is made into an array of strings. The
        #       array may contain repeat tokens
        #   TokenTable: SplitTable with repeated tokens removed per row.
        # Then to flatten the arrays, TokenTable has to be joined with itself.
        # See the sections 'Flattening Arrays' and 'Filtering Arrays' at
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/arrays
        sql = ('WITH SplitTable AS '
               '         (SELECT SPLIT({name}, \' \') as token_array FROM {table}), '
               '     TokenTable AS '
               '         (SELECT ARRAY(SELECT DISTINCT x '
               '                       FROM UNNEST(token_array) AS x) AS unique_tokens_per_row '
               '          FROM SplitTable) '
               'SELECT token, COUNT(token) as token_count '
               'FROM TokenTable '
               'CROSS JOIN UNNEST(TokenTable.unique_tokens_per_row) as token '
               'WHERE LENGTH(token) > 0 '
               'GROUP BY token '
               'ORDER BY token_count DESC, token ASC').format(name=col_name,
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
    elif transform_name in constant.NUMERIC_TRANSFORMS:
      # get min/max/average
      sql = ('SELECT max({name}) as max_value, min({name}) as min_value, '
             'avg({name}) as avg_value from {table}').format(name=col_name,
                                                             table=table_name)
      df = _execute_sql(sql, table)
      numerical_vocab_stats[col_name] = {'min': df.iloc[0]['min_value'],
                                         'max': df.iloc[0]['max_value'],
                                         'mean': df.iloc[0]['avg_value']}
    sys.stdout.write('done.\n')
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


def run_local_analysis(output_dir, csv_file_pattern, schema, inverted_features):
  """Use pandas to analyze csv files.

  Produces a stats file and vocab files.

  Args:
    output_dir: output folder
    csv_file_pattern: list of csv file paths, may contain wildcards
    schema: BQ schema list
    inverted_features: inverted_features dict

  Raises:
    ValueError: on unknown transfrorms/schemas
  """
  header = [column['name'] for column in schema]
  input_files = []
  for file_pattern in csv_file_pattern:
    input_files.extend(file_io.get_matching_files(file_pattern))

  # Make a copy of inverted_features and update the target transform to be
  # identity or one hot depending on the schema.
  inverted_features_target = copy.deepcopy(inverted_features)
  for name, transform_set in six.iteritems(inverted_features_target):
    if transform_set == set([constant.TARGET_TRANSFORM]):
      target_schema = next(col['type'].lower() for col in schema if col['name'] == name)
      if target_schema in constant.NUMERIC_SCHEMA:
        inverted_features_target[name] = {constant.IDENTITY_TRANSFORM}
      else:
        inverted_features_target[name] = {constant.ONE_HOT_TRANSFORM}

  # initialize the results
  def _init_numerical_results():
    return {'min': float('inf'),
            'max': float('-inf'),
            'count': 0,
            'sum': 0.0}
  numerical_results = collections.defaultdict(_init_numerical_results)
  vocabs = collections.defaultdict(lambda: collections.defaultdict(int))

  num_examples = 0
  # for each file, update the numerical stats from that file, and update the set
  # of unique labels.
  for input_file in input_files:
    sys.stdout.write('Analyzing file %s...' % input_file)
    sys.stdout.flush()
    with file_io.FileIO(input_file, 'r') as f:
      for line in csv.reader(f):
        if len(header) != len(line):
          raise ValueError('Schema has %d columns but a csv line only has %d columns.' %
                           (len(header), len(line)))
        parsed_line = dict(zip(header, line))
        num_examples += 1

        for col_name, transform_set in six.iteritems(inverted_features_target):
          # All transforms in transform_set require the same analysis. So look
          # at the first transform.
          transform_name = next(iter(transform_set))
          if transform_name in constant.TEXT_TRANSFORMS:
            split_strings = parsed_line[col_name].split(' ')

            # If a label is in the row N times, increase it's vocab count by 1.
            # This is needed for TFIDF, but it's also an interesting stat.
            for one_label in set(split_strings):
              # Filter out empty strings
              if one_label:
                vocabs[col_name][one_label] += 1
          elif transform_name in constant.CATEGORICAL_TRANSFORMS:
            if parsed_line[col_name]:
              vocabs[col_name][parsed_line[col_name]] += 1
          elif transform_name in constant.NUMERIC_TRANSFORMS:
            if not parsed_line[col_name].strip():
              continue

            numerical_results[col_name]['min'] = (
              min(numerical_results[col_name]['min'],
                  float(parsed_line[col_name])))
            numerical_results[col_name]['max'] = (
              max(numerical_results[col_name]['max'],
                  float(parsed_line[col_name])))
            numerical_results[col_name]['count'] += 1
            numerical_results[col_name]['sum'] += float(parsed_line[col_name])

    sys.stdout.write('done.\n')
    sys.stdout.flush()

  # Write the vocab files. Each label is on its own line.
  vocab_sizes = {}
  for name, label_count in six.iteritems(vocabs):
    # df is now:
    # label1,count
    # label2,count
    # ...
    # where label1 is the most frequent label, and label2 is the 2nd most, etc.
    df = pd.DataFrame([{'label': label, 'count': count}
                       for label, count in sorted(six.iteritems(label_count),
                                                  key=lambda x: x[1],
                                                  reverse=True)],
                      columns=['label', 'count'])
    csv_string = df.to_csv(index=False, header=False)

    file_io.write_string_to_file(
        os.path.join(output_dir, constant.VOCAB_ANALYSIS_FILE % name),
        csv_string)

    vocab_sizes[name] = {'vocab_size': len(label_count)}

  # Update numerical_results to just have min/min/mean
  for col_name in numerical_results:
    if float(numerical_results[col_name]['count']) == 0:
      raise ValueError('Column %s has a zero count' % col_name)
    mean = (numerical_results[col_name]['sum'] /
            float(numerical_results[col_name]['count']))
    del numerical_results[col_name]['sum']
    del numerical_results[col_name]['count']
    numerical_results[col_name]['mean'] = mean

  # Write the stats file.
  numerical_results.update(vocab_sizes)
  stats = {'column_stats': numerical_results, 'num_examples': num_examples}
  file_io.write_string_to_file(
      os.path.join(output_dir, constant.STATS_FILE),
      json.dumps(stats, indent=2, separators=(',', ': ')))


def check_schema_transforms_match(schema, inverted_features):
  """Checks that the transform and schema do not conflict.

  Args:
    schema: schema list
    inverted_features: inverted_features dict

  Raises:
    ValueError if transform cannot be applied given schema type.
  """
  num_target_transforms = 0

  for col_schema in schema:
    col_name = col_schema['name']
    col_type = col_schema['type'].lower()

    # Check each transform and schema are compatible
    if col_name in inverted_features:
      for transform_name in inverted_features[col_name]:
        if transform_name == constant.TARGET_TRANSFORM:
          num_target_transforms += 1
          continue

        elif col_type in constant.NUMERIC_SCHEMA:
          if transform_name not in constant.NUMERIC_TRANSFORMS:
            raise ValueError(
                'Transform %s not supported by schema %s' % (transform_name, col_type))
        elif col_type == constant.STRING_SCHEMA:
          if (transform_name not in constant.CATEGORICAL_TRANSFORMS + constant.TEXT_TRANSFORMS and
             transform_name != constant.IMAGE_TRANSFORM):
            raise ValueError(
                'Transform %s not supported by schema %s' % (transform_name, col_type))
        else:
          raise ValueError('Unsupported schema type %s' % col_type)

    # Check each transform is compatible for the same source column.
    # inverted_features[col_name] should belong to exactly 1 of the 5 groups.
    if col_name in inverted_features and 1 != (
      sum([inverted_features[col_name].issubset(set(constant.NUMERIC_TRANSFORMS)),
           inverted_features[col_name].issubset(set(constant.CATEGORICAL_TRANSFORMS)),
           inverted_features[col_name].issubset(set(constant.TEXT_TRANSFORMS)),
           inverted_features[col_name].issubset(set([constant.IMAGE_TRANSFORM])),
           inverted_features[col_name].issubset(set([constant.TARGET_TRANSFORM]))])):
      message = """
          The source column of a feature can only be used in multiple
          features within the same family of transforms. The familes are

          1) text transformations: %s
          2) categorical transformations: %s
          3) numerical transformations: %s
          4) image transformations: %s
          5) target transform: %s

          Any column can also be a key column.

          But column %s is used by transforms %s.
          """ % (str(constant.TEXT_TRANSFORMS),
                 str(constant.CATEGORICAL_TRANSFORMS),
                 str(constant.NUMERIC_TRANSFORMS),
                 constant.IMAGE_TRANSFORM,
                 constant.TARGET_TRANSFORM,
                 col_name,
                 str(inverted_features[col_name]))
      raise ValueError(message)

  if num_target_transforms != 1:
    raise ValueError('Must have exactly one target transform')


def expand_defaults(schema, features):
  """Add to features any default transformations.

  Not every column in the schema has an explicit feature transformation listed
  in the featurs file. For these columns, add a default transformation based on
  the schema's type. The features dict is modified by this function call.

  After this function call, every column in schema is used in a feature, and
  every feature uses a column in the schema.

  Args:
    schema: schema list
    features: features dict

  Raises:
    ValueError: if transform cannot be applied given schema type.
  """

  schema_names = [x['name'] for x in schema]

  # Add missing source columns
  for name, transform in six.iteritems(features):
    if 'source_column' not in transform:
      transform['source_column'] = name

  # Check source columns are in the schema and collect which are used.
  used_schema_columns = []
  for name, transform in six.iteritems(features):
    if transform['source_column'] not in schema_names:
      raise ValueError('source column %s is not in the schema for transform %s'
                       % (transform['source_column'], name))
    used_schema_columns.append(transform['source_column'])

  # Update default transformation based on schema.
  for col_schema in schema:
    schema_name = col_schema['name']
    schema_type = col_schema['type'].lower()

    if schema_type not in constant.NUMERIC_SCHEMA + [constant.STRING_SCHEMA]:
      raise ValueError(('Only the following schema types are supported: %s'
                        % ' '.join(constant.NUMERIC_SCHEMA + [constant.STRING_SCHEMA])))

    if schema_name not in used_schema_columns:
      # add the default transform to the features
      if schema_type in constant.NUMERIC_SCHEMA:
        features[schema_name] = {
            'transform': constant.DEFAULT_NUMERIC_TRANSFORM,
            'source_column': schema_name}
      elif schema_type == constant.STRING_SCHEMA:
        features[schema_name] = {
            'transform': constant.DEFAULT_CATEGORICAL_TRANSFORM,
            'source_column': schema_name}
      else:
        raise NotImplementedError('Unknown type %s' % schema_type)


# TODO(brandondutra): introduce the notion an analysis plan/classes if we
# support more complicated transforms like binning by quratiles.
def invert_features(features):
  """Make a dict in the form source column : set of transforms.

  Note that the key transform is removed.
  """
  inverted_features = collections.defaultdict(set)
  for transform in six.itervalues(features):
    source_column = transform['source_column']
    transform_name = transform['transform']
    if transform_name == constant.KEY_TRANSFORM:
      continue
    inverted_features[source_column].add(transform_name)

  return dict(inverted_features)  # convert from defaultdict to dict


def main(argv=None):
  args = parse_arguments(sys.argv if argv is None else argv)

  if args.csv_schema_file:
    schema = json.loads(
        file_io.read_file_to_string(args.csv_schema_file).decode())
  else:
    import google.datalab.bigquery as bq
    schema = bq.Table(args.bigquery_table).schema._bq_schema
  features = json.loads(
      file_io.read_file_to_string(args.features_file).decode())

  expand_defaults(schema, features)  # features are updated.
  inverted_features = invert_features(features)
  check_schema_transforms_match(schema, inverted_features)

  file_io.recursive_create_dir(args.output_dir)

  if args.cloud:
    run_cloud_analysis(
        output_dir=args.output_dir,
        csv_file_pattern=args.csv_file_pattern,
        bigquery_table=args.bigquery_table,
        schema=schema,
        inverted_features=inverted_features)
  else:
    run_local_analysis(
        output_dir=args.output_dir,
        csv_file_pattern=args.csv_file_pattern,
        schema=schema,
        inverted_features=inverted_features)

  # Save a copy of the schema and features in the output folder.
  file_io.write_string_to_file(
    os.path.join(args.output_dir, constant.SCHEMA_FILE),
    json.dumps(schema, indent=2))

  file_io.write_string_to_file(
    os.path.join(args.output_dir, constant.FEATURES_FILE),
    json.dumps(features, indent=2))


if __name__ == '__main__':
  main()
