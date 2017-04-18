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
import json
import os
import six
import sys


from tensorflow.python.lib.io import file_io

SCHEMA_FILE = 'schema.json'
NUMERICAL_ANALYSIS_FILE = 'stats.json'
CATEGORICAL_ANALYSIS_FILE = 'vocab_%s.csv'


def parse_arguments(argv):
  """Parse command line arguments.

  Args:
    argv: list of command line arguments, includeing programe name.

  Returns:
    An argparse Namespace object.

  Raises:
    ValueError: for bad parameters
  """
  parser = argparse.ArgumentParser(
      description='Runs Preprocessing on structured data.')
  parser.add_argument('--output-dir',
                      type=str,
                      required=True,
                      help='Google Cloud Storage which to place outputs.')

  parser.add_argument('--schema-file',
                      type=str,
                      required=False,
                      help=('BigQuery json schema file'))
  parser.add_argument('--input-file-pattern',
                      type=str,
                      required=False,
                      help='Input CSV file names. May contain a file pattern')

  # If using bigquery table
  # TODO(brandondutra): maybe also support an sql input, so the table can be
  # ad-hoc.
  parser.add_argument('--bigquery-table',
                      type=str,
                      required=False,
                      help=('project:dataset.table_name'))

  args = parser.parse_args(args=argv[1:])

  if not args.output_dir.startswith('gs://'):
    raise ValueError('--output-dir must point to a location on GCS')

  if args.bigquery_table:
    if args.schema_file or args.input_file_pattern:
      raise ValueError('If using --bigquery-table, then --schema-file and '
                       '--input-file-pattern, '
                       'are not needed.')
  else:
    if not args.schema_file or not args.input_file_pattern:
      raise ValueError('If not using --bigquery-table, then --schema-file and '
                       '--input-file-pattern '
                       'are required.')

    if not args.input_file_pattern.startswith('gs://'):
      raise ValueError('--input-file-pattern must point to files on GCS')

  return args


def parse_table_name(bigquery_table):
  """Giving a string a:b.c, returns b.c.

  Args:
    bigquery_table: full table name project_id:dataset:table

  Returns:
    dataset:table

  Raises:
    ValueError: if a, b, or c contain the character ':'.
  """

  id_name = bigquery_table.split(':')
  if len(id_name) != 2:
    raise ValueError('Bigquery table name should be in the form '
                     'project_id:dataset.table_name. Got %s' % bigquery_table)
  return id_name[1]


def run_numerical_analysis(table, schema_list, args):
  """Find min/max values for the numerical columns and writes a json file.

  Args:
    table: Reference to FederatedTable (if bigquery_table is false) or a
        regular Table (otherwise)
    schema_list: Bigquery schema json object
    args: the command line args
  """
  import google.datalab.bigquery as bq

  # Get list of numerical columns.
  numerical_columns = []
  for col_schema in schema_list:
    col_type = col_schema['type'].lower()
    if col_type == 'integer' or col_type == 'float':
      numerical_columns.append(col_schema['name'])

  # Run the numerical analysis
  if numerical_columns:
    sys.stdout.write('Running numerical analysis...')
    max_min = [
        ('max({name}) as max_{name}, '
         'min({name}) as min_{name}, '
         'avg({name}) as avg_{name} ').format(name=name)
        for name in numerical_columns]
    if args.bigquery_table:
      sql = 'SELECT  %s from `%s`' % (', '.join(max_min), parse_table_name(args.bigquery_table))
      numerical_results = bq.Query(sql).execute().result().to_dataframe()
    else:
      sql = 'SELECT  %s from csv_table' % ', '.join(max_min)
      query = bq.Query(sql, data_sources={'csv_table': table})
      numerical_results = query.execute().result().to_dataframe()

    # Convert the numerical results to a json file.
    results_dict = {}
    for name in numerical_columns:
      results_dict[name] = {'max': numerical_results.iloc[0]['max_%s' % name],
                            'min': numerical_results.iloc[0]['min_%s' % name],
                            'mean': numerical_results.iloc[0]['avg_%s' % name]}

    file_io.write_string_to_file(
        os.path.join(args.output_dir, NUMERICAL_ANALYSIS_FILE),
        json.dumps(results_dict, indent=2, separators=(',', ': ')))

    sys.stdout.write('done.\n')


def run_categorical_analysis(table, schema_list, args):
  """Find vocab values for the categorical columns and writes a csv file.

  The vocab files are in the from
  label1
  label2
  label3
  ...

  Args:
    table: Reference to FederatedTable (if bigquery_table is false) or a
        regular Table (otherwise)
    schema_list: Bigquery schema json object
    args: the command line args
  """
  import google.datalab.bigquery as bq

  # Get list of categorical columns.
  categorical_columns = []
  for col_schema in schema_list:
    col_type = col_schema['type'].lower()
    if col_type == 'string':
      categorical_columns.append(col_schema['name'])

  if categorical_columns:
    sys.stdout.write('Running categorical analysis...')
    for name in categorical_columns:
      if args.bigquery_table:
        table_name = parse_table_name(args.bigquery_table)
      else:
        table_name = 'table_name'

      sql = """
            SELECT
              {name}
            FROM
              {table}
            WHERE
              {name} IS NOT NULL
            GROUP BY
              {name}
            ORDER BY
              {name}
      """.format(name=name, table=table_name)
      out_file = os.path.join(args.output_dir,
                              CATEGORICAL_ANALYSIS_FILE % name)

      # extract_async seems to have a bug and sometimes hangs. So get the
      # results direclty.
      if args.bigquery_table:
        df = bq.Query(sql).execute().result().to_dataframe()
      else:
        query = bq.Query(sql, data_sources={'table_name': table})
        df = query.execute().result().to_dataframe()

      # Write the results to a file.
      string_buff = six.StringIO()
      df.to_csv(string_buff, index=False, header=False)
      file_io.write_string_to_file(out_file, string_buff.getvalue())

    sys.stdout.write('done.\n')


def run_analysis(args):
  """Builds an analysis file for training.

  Uses BiqQuery tables to do the analysis.

  Args:
    args: command line args

  Raises:
    ValueError if schema contains unknown types.
  """
  import google.datalab.bigquery as bq
  if args.bigquery_table:
    table = bq.Table(args.bigquery_table)
    schema_list = table.schema._bq_schema
  else:
    schema_list = json.loads(
        file_io.read_file_to_string(args.schema_file).decode())
    table = bq.ExternalDataSource(
        source=args.input_file_pattern,
        schema=bq.Schema(schema_list))

  # Check the schema is supported.
  for col_schema in schema_list:
    col_type = col_schema['type'].lower()
    if col_type != 'string' and col_type != 'integer' and col_type != 'float':
      raise ValueError('Schema contains an unsupported type %s.' % col_type)

  run_numerical_analysis(table, schema_list, args)
  run_categorical_analysis(table, schema_list, args)

  # Save a copy of the schema to the output location.
  file_io.write_string_to_file(
      os.path.join(args.output_dir, SCHEMA_FILE),
      json.dumps(schema_list, indent=2, separators=(',', ': ')))


def main(argv=None):
  args = parse_arguments(sys.argv if argv is None else argv)
  run_analysis(args)


if __name__ == '__main__':
  main()
