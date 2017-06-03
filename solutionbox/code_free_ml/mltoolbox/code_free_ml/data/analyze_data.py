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
import base64
import collections
import cStringIO
import csv
import json
import os
from PIL import Image
import sys
import six
import textwrap

from tensorflow.python.lib.io import file_io

# Files
SCHEMA_FILE = 'schema.json'
FEATURES_FILE = 'features.json'
STATS_FILE = 'stats.json'
VOCAB_ANALYSIS_FILE = 'vocab_%s.csv'

TRANSFORMED_METADATA_DIR = 'transformed_metadata'
RAW_METADATA_DIR = 'raw_metadata'
TRANSFORM_FN_DIR = 'transform_fn'

# Individual transforms
IDENTITY_TRANSFORM = 'identity'
SCALE_TRANSFORM = 'scale'
ONE_HOT_TRANSFORM = 'one_hot'
EMBEDDING_TRANSFROM = 'embedding'
BOW_TRANSFORM = 'bag_of_words'
TFIDF_TRANSFORM = 'tfidf'
KEY_TRANSFORM = 'key'
TARGET_TRANSFORM = 'target'
IMAGE_TRANSFORM = 'image_to_vec'

# Transform collections
NUMERIC_TRANSFORMS = [IDENTITY_TRANSFORM, SCALE_TRANSFORM]
CATEGORICAL_TRANSFORMS = [ONE_HOT_TRANSFORM, EMBEDDING_TRANSFROM]
TEXT_TRANSFORMS = [BOW_TRANSFORM, TFIDF_TRANSFORM]

# If the features file is missing transforms, apply these.
DEFAULT_NUMERIC_TRANSFORM = IDENTITY_TRANSFORM
DEFAULT_CATEGORICAL_TRANSFORM = ONE_HOT_TRANSFORM

# Schema values
INTEGER_SCHEMA = 'integer'
FLOAT_SCHEMA = 'float'
STRING_SCHEMA = 'string'
NUMERIC_SCHEMA = [INTEGER_SCHEMA, FLOAT_SCHEMA]

# Inception Checkpoint
INCEPTION_V3_CHECKPOINT = 'gs://cloud-ml-data/img/flower_photos/inception_v3_2016_08_28.ckpt'
INCEPTION_EXCLUDED_VARIABLES = ['InceptionV3/AuxLogits', 'InceptionV3/Logits', 'global_step']

_img_buf = cStringIO.StringIO()
Image.new('RGB', (16, 16)).save(_img_buf, 'jpeg')
IMAGE_DEFAULT_STRING = base64.urlsafe_b64encode(_img_buf.getvalue())


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
                "column_name_4": {"transform": "key"},
             }

             The format of the dict is `name`: `transform-dict` where the
             `name` must be a column name from the schema file. A list of
             supported `transform-dict`s is below:

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
                      type=str,
                      required=True,
                      help='GCS or local folder')

  # CSV inputs
  parser.add_argument('--csv-file-pattern',
                      type=str,
                      required=False,
                      help=('Input CSV file names. May contain a file pattern. '
                            'File prefix must include absolute file path.'))
  parser.add_argument('--csv-schema-file',
                      type=str,
                      required=False,
                      help=('BigQuery json schema file path'))

  # If using bigquery table
  parser.add_argument('--bigquery-table',
                      type=str,
                      required=False,
                      help=('project.dataset.table_name'))

  parser.add_argument('--features-file',
                      type=str,
                      required=True,
                      help='Features file path')

  args = parser.parse_args(args=argv[1:])

  if args.cloud:
    if not args.output_dir.startswith('gs://'):
      raise ValueError('--output-dir must point to a location on GCS')
    if args.csv_file_pattern and not args.csv_file_pattern.startswith('gs://'):
      raise ValueError('--csv-file-pattern must point to a location on GCS')
    if args.csv_schema_file and not args.csv_schema_file.startswith('gs://'):
      raise ValueError('--csv-schema-file must point to a location on GCS')

  if not ((args.bigquery_table and args.csv_file_pattern is None and
           args.csv_schema_file is None) or
          (args.bigquery_table is None and args.csv_file_pattern and
           args.csv_schema_file)):
    raise ValueError('either --csv-schema-file and --csv-file-pattern must both'
                     ' be set or just --bigquery-table is set')

  return args


def run_cloud_analysis(output_dir, csv_file_pattern, bigquery_table, schema,
                       features):
  """Use BigQuery to analyze input date.

  Only one of csv_file_pattern or bigquery_table should be non-None.

  Args:
    output_dir: output folder
    csv_file_pattern: csv file path, may contain wildcards
    bigquery_table: project_id.dataset_name.table_name
    schema: schema list
    features: features dict
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

  numerical_vocab_stats = {}

  for col_schema in schema:
    col_name = col_schema['name']
    col_type = col_schema['type'].lower()
    transform = features[col_name]['transform']

    # Map the target transfrom into one_hot or identity.
    if transform == TARGET_TRANSFORM:
      if col_type == STRING_SCHEMA:
        transform = ONE_HOT_TRANSFORM
      elif col_type in NUMERIC_SCHEMA:
        transform = IDENTITY_TRANSFORM
      else:
        raise ValueError('Unknown schema type')

    if transform in (TEXT_TRANSFORMS + CATEGORICAL_TRANSFORMS):
      if transform in TEXT_TRANSFORMS:
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
      string_buff = six.StringIO()
      df.to_csv(string_buff, index=False, header=False)
      file_io.write_string_to_file(
          os.path.join(output_dir, VOCAB_ANALYSIS_FILE % col_name),
          string_buff.getvalue())
      numerical_vocab_stats[col_name] = {'vocab_size': len(df)}

      # free memeory
      del string_buff
      del df
    elif transform in NUMERIC_TRANSFORMS:
      # get min/max/average
      sql = ('SELECT max({name}) as max_value, min({name}) as min_value, '
             'avg({name}) as avg_value from {table}').format(name=col_name,
                                                             table=table_name)
      df = _execute_sql(sql, table)
      numerical_vocab_stats[col_name] = {'min': df.iloc[0]['min_value'],
                                         'max': df.iloc[0]['max_value'],
                                         'mean': df.iloc[0]['avg_value']}
    elif transform == IMAGE_TRANSFORM:
      pass
    elif transform == KEY_TRANSFORM:
      pass
    else:
      raise ValueError('Unknown transform %s' % transform)

  # get num examples
  sql = 'SELECT count(*) as num_examples from {table}'.format(table=table_name)
  df = _execute_sql(sql, table)
  num_examples = df.iloc[0]['num_examples']

  # Write the stats file.
  stats = {'column_stats': numerical_vocab_stats, 'num_examples': num_examples}
  file_io.write_string_to_file(
      os.path.join(output_dir, STATS_FILE),
      json.dumps(stats, indent=2, separators=(',', ': ')))


def run_local_analysis(output_dir, csv_file_pattern, schema, features):
  """Use pandas to analyze csv files.

  Produces a stats file and vocab files.

  Args:
    output_dir: output folder
    csv_file_pattern: string, may contain wildcards
    schema: BQ schema list
    features: features dict

  Raises:
    ValueError: on unknown transfrorms/schemas
  """
  header = [column['name'] for column in schema]
  input_files = file_io.get_matching_files(csv_file_pattern)

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
    with file_io.FileIO(input_file, 'r') as f:
      for line in csv.reader(f):
        if len(header) != len(line):
          raise ValueError('Schema has %d columns but a csv line only has %d columns.' %
                           (len(header), len(line)))
        parsed_line = dict(zip(header, line))
        num_examples += 1

        for col_schema in schema:
          col_name = col_schema['name']
          col_type = col_schema['type'].lower()
          transform = features[col_name]['transform']

          # Map the target transfrom into one_hot or identity.
          if transform == TARGET_TRANSFORM:
            if col_type == STRING_SCHEMA:
              transform = ONE_HOT_TRANSFORM
            elif col_type in NUMERIC_SCHEMA:
              transform = IDENTITY_TRANSFORM
            else:
              raise ValueError('Unknown schema type')

          if transform in TEXT_TRANSFORMS:
            split_strings = parsed_line[col_name].split(' ')

            # If a label is in the row N times, increase it's vocab count by 1.
            # This is needed for TFIDF, but it's also an interesting stat.
            for one_label in set(split_strings):
              # Filter out empty strings
              if one_label:
                vocabs[col_name][one_label] += 1
          elif transform in CATEGORICAL_TRANSFORMS:
            if parsed_line[col_name]:
              vocabs[col_name][parsed_line[col_name]] += 1
          elif transform in NUMERIC_TRANSFORMS:
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
          elif transform == IMAGE_TRANSFORM:
            pass
          elif transform == KEY_TRANSFORM:
            pass
          else:
            raise ValueError('Unknown transform %s' % transform)

  # Write the vocab files. Each label is on its own line.
  vocab_sizes = {}
  for name, label_count in six.iteritems(vocabs):
    # Labels is now the string:
    # label1,count
    # label2,count
    # ...
    # where label1 is the most frequent label, and label2 is the 2nd most, etc.
    labels = '\n'.join(['%s,%d' % (label, count)
                        for label, count in sorted(six.iteritems(label_count),
                                                   key=lambda x: x[1],
                                                   reverse=True)])
    file_io.write_string_to_file(
        os.path.join(output_dir, VOCAB_ANALYSIS_FILE % name),
        labels)

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
      os.path.join(output_dir, STATS_FILE),
      json.dumps(stats, indent=2, separators=(',', ': ')))


def check_schema_transforms_match(schema, features):
  """Checks that the transform and schema do not conflict.

  Args:
    schema: schema file
    features: features file

  Raises:
    ValueError if transform cannot be applied given schema type.
  """
  num_target_transforms = 0

  for col_schema in schema:
    col_name = col_schema['name']
    col_type = col_schema['type'].lower()

    transform = features[col_name]['transform']
    if transform == KEY_TRANSFORM:
      continue
    elif transform == TARGET_TRANSFORM:
      num_target_transforms += 1
      continue

    if col_type in NUMERIC_SCHEMA:
      if transform not in NUMERIC_TRANSFORMS:
        raise ValueError(
            'Transform %s not supported by schema %s' % (transform, col_type))
    elif col_type == STRING_SCHEMA:
      if (transform not in CATEGORICAL_TRANSFORMS + TEXT_TRANSFORMS and
         transform != IMAGE_TRANSFORM):
        raise ValueError(
            'Transform %s not supported by schema %s' % (transform, col_type))
    else:
      raise ValueError('Unsupported schema type %s' % col_type)

  if num_target_transforms != 1:
    raise ValueError('Must have exactly one target transform')


def expand_defaults(schema, features):
  """Add to features any default transformations.

  Not every column in the schema has an explicit feature transformation listed
  in the featurs file. For these columns, add a default transformation based on
  the schema's type. The features dict is modified by this function call.

  Args:
    schema: schema list
    features: features dict

  Raises:
    ValueError: if transform cannot be applied given schema type.
  """

  schema_names = [x['name'] for x in schema]

  for source_column in six.iterkeys(features):
    if source_column not in schema_names:
      raise ValueError('source column %s is not in the schema' % source_column)

  # Update default transformation based on schema.
  for col_schema in schema:
    schema_name = col_schema['name']
    schema_type = col_schema['type'].lower()

    if schema_type not in NUMERIC_SCHEMA + [STRING_SCHEMA]:
      raise ValueError(('Only the following schema types are supported: %s'
                        % ' '.join(NUMERIC_SCHEMA + [STRING_SCHEMA])))

    if schema_name not in six.iterkeys(features):
      # add the default transform to the features
      if schema_type in NUMERIC_SCHEMA:
        features[schema_name] = {'transform': DEFAULT_NUMERIC_TRANSFORM}
      elif schema_type == STRING_SCHEMA:
        features[schema_name] = {'transform': DEFAULT_CATEGORICAL_TRANSFORM}
      else:
        raise NotImplementedError('Unknown type %s' % schema_type)


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
  check_schema_transforms_match(schema, features)

  file_io.recursive_create_dir(args.output_dir)

  if args.cloud:
    run_cloud_analysis(
        output_dir=args.output_dir,
        csv_file_pattern=args.csv_file_pattern,
        bigquery_table=args.bigquery_table,
        schema=schema,
        features=features)
  else:
    run_local_analysis(
        output_dir=args.output_dir,
        csv_file_pattern=args.csv_file_pattern,
        schema=schema,
        features=features)

  # Save a copy of the schema and features in the output folder.
  file_io.write_string_to_file(
    os.path.join(args.output_dir, SCHEMA_FILE),
    json.dumps(schema, indent=2))

  file_io.write_string_to_file(
    os.path.join(args.output_dir, FEATURES_FILE),
    json.dumps(features, indent=2))


if __name__ == '__main__':
  main()
