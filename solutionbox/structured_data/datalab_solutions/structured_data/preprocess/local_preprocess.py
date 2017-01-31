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

import argparse
import collections
import os
import sys
import json
import pandas as pd
import google.cloud.ml as ml

from StringIO import StringIO

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
      description='Runs Preprocessing on structured CSV data.')
  parser.add_argument('--input_file_pattern',
                      type=str,
                      required=True,
                      help='Input CSV file names. May contain a file pattern')
  parser.add_argument('--output_dir',
                      type=str,
                      required=True,
                      help='Google Cloud Storage which to place outputs.')
  parser.add_argument('--schema_file',
                      type=str,
                      required=True,
                      help=('BigQuery json schema file'))
  parser.add_argument('--input_feature_types',
                      type=str,
                      required=True,
                      help=('Json file containing feature types'))

  args = parser.parse_args(args=argv[1:])

  # Make sure the output folder exists if local folder.
  ml.util._file.create_directory(args.output_dir)

  return args

def run_numerical_categorical_analysis(args, feature_types, schema_list):
  header = [column['name'] for column in schema_list]
  input_files = ml.util._file.glob_files(args.input_file_pattern)

  # initialize numerical_results
  numerical_results = {}
  for name, config in feature_types.iteritems():
    if config['type'] == 'numerical':
      numerical_results[name] = {'min': float('inf'), 'max': float('-inf')}

  # initialize categorical_results
  categorical_results = collections.defaultdict(set)

  # for each file, update the min/max values from that file, and update the set
  # of unique labels.
  for input_file in input_files:
    df = pd.read_csv(StringIO(ml.util._file.load_file(input_file)),
                     header=None,
                     names=header)

    for name, config in feature_types.iteritems():
      # Update numerical analsysis
      if config['type'] == 'numerical':
        numerical_results[name]['min'] = min(numerical_results[name]['min'], 
                                             df[name].min())
        numerical_results[name]['max'] = max(numerical_results[name]['max'],
                                             df[name].max())
      elif config['type'] == 'categorical':
        # Update categorical analsysis
        categorical_results[name].update(set(df[name].unique()))
      elif config['type'] == 'key':
        pass
      else:
        raise ValueError('Unknown type %s in input features' % config['type'])

  # Write the numerical_results to a json file.
  with ml.util._file.open_local_or_gcs(
        os.path.join(args.output_dir, NUMERICAL_ANALYSIS), 'w') as f:
    f.write(json.dumps(numerical_results, indent=2, separators=(',', ': ')))

  # Write the vocab files.
  for name, unique_labels in categorical_results.iteritems():
    with ml.util._file.open_local_or_gcs(
        os.path.join(args.output_dir, CATEGORICAL_ANALYSIS % name),
        'w') as f:
      f.write('index,%s\n' % name)
      df = pd.DataFrame(list(categorical_results[name]))
      f.write(df.to_csv(None, header=False))


def run_analysis(args):
  """Builds an analysis file for training.

  Reads all the data into memery with Pandas.
  """
  # Read the schema and input feature types
  with ml.util._file.open_local_or_gcs(args.schema_file, 'r') as f:
    schema_list = json.loads(f.read())
  with ml.util._file.open_local_or_gcs(args.input_feature_types, 'r') as f:
    feature_types = json.loads(f.read())

  run_numerical_categorical_analysis(args, feature_types, schema_list)

  # Also save a copy of the schema/input types in the output folder.
  ml.util._file.copy_file(args.input_feature_types,
                          os.path.join(args.output_dir, INPUT_FEATURES_FILE))
  ml.util._file.copy_file(args.schema_file,
                          os.path.join(args.output_dir, SCHEMA_FILE))


def main(argv=None):
  args = parse_arguments(sys.argv if argv is None else argv)
  run_analysis(args)


if __name__ == '__main__':
  main()
