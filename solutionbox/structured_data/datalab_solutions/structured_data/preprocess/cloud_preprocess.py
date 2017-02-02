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
import os
import sys
import json

import google.cloud.ml as ml
import datalab.bigquery as bq

INPUT_FEATURES_FILE = 'input_features.json'
SCHEMA_FILE = 'schema.json'

NUMERICAL_ANALYSIS = 'numerical_analysis.json'
CATEGORICAL_ANALYSIS = 'vocab_%s.csv'


def parse_arguments(argv):
  """Parse command line arguments.

  Args:
    argv: list of command line arguments, includeing programe name.

  Returns:
    An argparse Namespace object.
  """
  parser = argparse.ArgumentParser(
      description='Runs Preprocessing on structured data.')
  parser.add_argument('--output_dir',
                      type=str,
                      required=True,
                      help='Google Cloud Storage which to place outputs.')
  parser.add_argument('--input_feature_types',
                      type=str,
                      required=True,
                      help=('Json file containing feature types'))

  # If using csv files.
  parser.add_argument('--schema_file',
                      type=str,
                      required=False,
                      help=('BigQuery json schema file'))  
  parser.add_argument('--input_file_pattern',
                      type=str,
                      required=False,
                      help='Input CSV file names. May contain a file pattern') 

  # If using bigquery table
  parser.add_argument('--bigquery_table',
                      type=str,
                      required=False,
                      help=('project:dataset.table_name'))

  args = parser.parse_args(args=argv[1:])
  print(args)

  if not args.output_dir.startswith('gs://'):
    raise ValueError('--output_dir must point to a location on GCS')

  if args.bigquery_table:
    if args.schema_file or args.input_file_pattern :
      raise ValueError('If using --bigquery_table, then --schema_file and '
                       '--input_file_pattern, '
                       'are not needed.')
  else:
    if not args.schema_file or not args.input_file_pattern :
      raise ValueError('If not using --bigquery_table, then --schema_file and '
                       '--input_file_pattern '
                       'are required.')

    if not args.input_file_pattern.startswith('gs://'):
      raise ValueError('--input_file_pattern must point to files on GCS')

  return args

def parse_table_name(bigquery_table):
  """Giving a string a:b.c, returns 'b.c'"""

  id_name = bigquery_table.split(':')
  if len(id_name) != 2:
    raise ValueError('Bigquery table name should be in the form '
                     'project_id:dataset.table_name. Got %s' % bigquery_table)
  return id_name[1]

def run_numerical_analysis(table, args, feature_types):
  """Find min/max values for the numerical columns and writes a json file.

  Args:
    table: Reference to FederatedTable if bigquery_table is false.
    args: the parseargs
    feature_types: python object of the feature types file.
  """
  # Get list of numerical columns.
  numerical_columns = []
  for name, config in feature_types.iteritems():
    if config['type'] == 'numerical':
      numerical_columns.append(name)

  # Run the numerical analysis
  if numerical_columns:
    sys.stdout.write('Running numerical analysis...')
    max_min = [
        "max({name}) as max_{name}, min({name}) as min_{name}".format(name=name)
        for name in numerical_columns]
    if args.bigquery_table:
      sql = "SELECT  %s from %s" % (', '.join(max_min), 
                                    parse_table_name(args.bigquery_table))
      numerical_results = bq.Query(sql).to_dataframe()
    else:
      sql = "SELECT  %s from csv_table" % ', '.join(max_min)
      query = bq.Query(sql, data_sources={'csv_table': table})
      numerical_results = query.to_dataframe()

    # Convert the numerical results to a json file.
    results_dict = {}
    for name in numerical_columns:
      results_dict[name] = {'max': numerical_results.iloc[0]['max_%s' % name],
                            'min': numerical_results.iloc[0]['min_%s' % name]}
    with ml.util._file.open_local_or_gcs(
        os.path.join(args.output_dir, NUMERICAL_ANALYSIS),
        'w') as f:
      f.write(json.dumps(results_dict, indent=2, separators=(',', ': ')))

    sys.stdout.write('done.\n')


def run_categorical_analysis(table, args, feature_types):
  """Find vocab values for the categorical columns and writes a csv file.

  The vocab files are in the from
  index,categorical_column_name
  0,'abc'
  1,'def'
  2,'ghi'
  ...

  Args:
    table: Reference to FederatedTable if bigquery_table is false.
    args: the parseargs
    feature_types: python object of the feature types file.
  """  
  
  categorical_columns = []
  for name, config in feature_types.iteritems():
    if config['type'] == 'categorical':
      categorical_columns.append(name)

  jobs = []
  if categorical_columns:
    sys.stdout.write('Running categorical analysis...')
    for name in categorical_columns:
      if args.bigquery_table:
        table_name = parse_table_name(args.bigquery_table)
      else:
        table_name = 'table_name'

      # BQ does not have a distinct function, or a row number that starts at 0.
      sql = """
          SELECT
            rn -1 AS index,
            {name}
          FROM (
            SELECT
              {name},
              ROW_NUMBER() OVER() AS rn
            FROM
              {table}
            WHERE
              {name} IS NOT NULL
            GROUP BY
              {name}
          )""".format(name=name, table=table_name)
      out_file = os.path.join(args.output_dir, CATEGORICAL_ANALYSIS % name)

      if args.bigquery_table:
        jobs.append(bq.Query(sql).extract_async(out_file))
      else:
        query = bq.Query(sql, data_sources={table_name: table})
        jobs.append(query.extract_async(out_file))

    for job in jobs:
      job.wait()

    sys.stdout.write('done.\n')

def run_analysis(args):
  """Builds an analysis file for training.

  Uses BiqQuery tables to do the analysis.
  """

  if args.bigquery_table:
    table = bq.Table(args.bigquery_table)
  else:
    with ml.util._file.open_local_or_gcs(args.schema_file, 'r') as f:
      schema_list = json.loads(f.read())    
    table = bq.FederatedTable().from_storage(
       source=args.input_file_pattern, 
       source_format='csv',
       ignore_unknown_values=False, 
       max_bad_records=0, 
       compressed=False,
       schema=bq.Schema(schema_list))

  with ml.util._file.open_local_or_gcs(args.input_feature_types, 'r') as f:
    feature_types = json.loads(f.read())
  
  run_numerical_analysis(table, args, feature_types)
  run_categorical_analysis(table, args, feature_types)

  # Save a copy of the input types to the output location.
  ml.util._file.copy_file(args.input_feature_types,
                          os.path.join(args.output_dir, INPUT_FEATURES_FILE))  

  # Save a copy of the schema to the output location.
  if args.schema_file:
    ml.util._file.copy_file(args.schema_file,
                           os.path.join(args.output_dir, SCHEMA_FILE))  
  else:
    output_schema = os.path.join(args.output_dir, SCHEMA_FILE)
    with ml.util._file.open_local_or_gcs(output_schema, 'w') as f:
      f.write(json.dumps(table.schema._bq_schema, indent=2, 
                         separators=(',', ': ')))



# def run_analysisxxx(args):
#   """Builds an analysis file for training.

#   Uses BiqQuery federated tables to do numerical analysis.
#   """
#   #TODO(brandondutra): should we also take a "numerical/categorical column" list
#   # and build vocabs for the categorical columns?

#   # Make a table from the input csv fiels.
#   with ml.util._file.open_local_or_gcs(args.schema_file, 'r') as f:
#     schema_list = json.loads(f.read())
#   schema = bq.Schema(schema_list)

#   table = bq.FederatedTable().from_storage(
#       source=args.input_file_pattern, source_format='csv',
#       ignore_unknown_values=False, max_bad_records=0, compressed=False,
#       schema=schema)

#   # For all the numerical columns, find the min/max and save the results to a
#   # file. Do this even if a numerial column is a key or target, as we do not
#   # know the meaning of the columns at preprocessing time.
#   numerical_columns = []
#   for column in schema_list:
#     column_type = column['type'].lower()
#     if column_type == 'float' or column_type == 'integer':
#       numerical_columns.append(column['name'])
#     elif column_type == 'string':
#       pass
#     else:
#       raise ValueError('only float, integer and string column types are '
#                        'supported in the schema')

#   # Run the numerical analysis
#   max_min = [
#       "max({name}) as max_{name}, min({name}) as min_{name}".format(name=name)
#       for name in numerical_columns]
#   sql = 'SELECT ' + ', '.join(max_min) + ' from csvtable'
#   numerical_query = bq.Query(sql, data_sources={'csvtable': table})
#   numerical_results = numerical_query.to_dataframe()

#   # Convert the numerical results to a json file.
#   results_dict = {}
#   for name in numerical_columns:
#     results_dict[name] = {'max': numerical_results.iloc[0]['max_%s' % name],
#                           'min': numerical_results.iloc[0]['min_%s' % name]}
#   with ml.util._file.open_local_or_gcs(os.path.join(args.output_dir, 'numerical_analysis.json'), 'w') as f:
#     f.write(json.dumps(results_dict, indent=2, separators=(',', ': ')))


#   # Also save a copy of the schema in the preprocess output folder.
#   ml.util._file.copy_file(args.schema_file,
#                           os.path.join(args.output_dir, 'schema.json'))


def main(argv=None):
  args = parse_arguments(sys.argv if argv is None else argv)
  run_analysis(args)


if __name__ == '__main__':
  main()
