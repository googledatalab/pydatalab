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
import datetime
import os
import random
import sys
import json

import apache_beam as beam

import google.cloud.ml as ml
import google.cloud.ml.features as features
import google.cloud.ml.io as io


def parse_arguments(argv):
  """Parse command line arguments.

  Args:
    argv: list of command line arguments, includeing programe name.

  Returns:
    An argparse Namespace object.
  """
  parser = argparse.ArgumentParser(
      description='Runs Preprocessing on structured CSV data.')
  parser.add_argument('--project_id',
                      help='The project to which the job will be submitted.')
  parser.add_argument('--cloud', action='store_true',
                      help='Run preprocessing on the cloud.')
  parser.add_argument('--input_file_path',
                      type=str,
                      required=True,
                      help='Input files names. May contain file patterns')
  parser.add_argument('--train_percent',
                      default=80,
                      type=int,
                      help='Percent of input data for training dataset.')
  parser.add_argument('--eval_percent',
                      default=10,
                      type=int,
                      help='Percent of input data for eval dataset.')
  parser.add_argument('--test_percent',
                      default=10,
                      type=int,
                      help='Percent of input data for test dataset.')
  parser.add_argument('--output_dir',
                      type=str,    
                      required=True,
                      help=('Google Cloud Storage or Local directory in which '
                            'to place outputs.'))
  parser.add_argument('--schema_file',
                      type=str,    
                      required=True,
                      help=('File describing the schema of each column in the '
                            'csv data files.'))
  parser.add_argument('--job_name',
                      type=str,
                      help=('If using --cloud, the job name as listed in'
                            'Dataflow on GCE. Defaults to '
                            'structured-data-[timestamp['))


  args = parser.parse_args(args=argv[1:])

  # Only the train set should not be empty.
  if (args.train_percent <= 0 or args.eval_percent < 0
      or args.test_percent < 0
      or args.train_percent + args.eval_percent + args.test_percent != 100):
    print('Error: train, eval, and test percents do not make sense')
    sys.exit(1)

  if args.cloud and not args.project_id:
    print('Error: Using --cloud but --project_id is missing.')
    sys.exit(1)

  # args.job_name will not be used unless --cloud is used.
  if not args.job_name:
    args.job_name = ('structured-data-' + 
                     datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

  return args


def load_and_check_config(schema_file_path):
  """Checks the sschema file is well formatted."""

  try:
    json_str = ml.util._file.load_file(schema_file_path)
    config = json.loads(json_str) 
  except:
    print('ERROR reading schema file.')
    sys.exit(1)

  model_columns = (config.get('numerical_columns', []) 
                   + config.get('categorical_columns', []))
  if config['target_column'] not in model_columns:
    print('ERROR: target not listed as a numerical or categorial column.')
    sys.exit(1)

  if set(config['column_names']) != set(model_columns + [config['key_column']]):
    print('ERROR: column_names do not match what was listed other fields')
    sys.exit(1)

  if set(config['numerical_columns']) & set(config['categorical_columns']):
    print('ERROR: numerical_columns and categorical_columns must be disjoint.')
    sys.exit(1)

  if config['key_column'] in model_columns:
    print('ERROR: kye_column should not be listed in numerical_columns or categorical_columns')
    sys.exit(1)

  return config


def preprocessing_features(args):

  # Read the config file.
  config = load_and_check_config(args.schema_file)

  column_names = config['column_names']

  feature_set = {}

  # Extract key feature
  feature_set[config['key_column']] = features.key(config['key_column'])

  # Extract target feature
  target_name = config['target_column']
  key_name = config['key_column']
  if target_name in config.get('numerical_columns', []):
    feature_set[target_name] = features.target(target_name).continuous()
  else:
    feature_set[target_name] = features.target(target_name).discrete()


  # Extract numeric features
  for name in config.get('numerical_columns', []):
    if name == target_name or name == key_name:
      continue
    # apply identity to all numerical features.
    default = config.get('defaults', {}).get(name, None)
    feature_set[name] = features.numeric(name, default=default).identity()

  # Extract categorical features
  for name in config.get('categorical_columns', []):
    if name == target_name or name == key_name:
      continue    
    # apply sparse transform to all categorical features.
    default = config.get('defaults', {}).get(name, None)
    feature_set[name] = features.categorical(
          name, 
          default=default,
          frequency_threshold=1).sparse(use_counts=True)

  return feature_set, column_names


def preprocess(pipeline, feature_set, column_names, input_file_path,
               train_percent, eval_percent, test_percent, output_dir):
  """Builds the preprocessing Dataflow pipeline.

  The input files are split into a training, eval and test sets, and the SDK
  preprocessing is applied to each set. The analysis state is only applied on
  the training set.
  """
  coder_with_target = io.CsvCoder.from_feature_set(feature_set,
                                                   column_names)

  # Split the data into training, evaluate, and test sets
  random.seed(12798)
  def _partition_fn(row_unused, num_partitions_unused):  # pylint: disable=unused-argument
    rand_num = random.randint(1, 100)
    if rand_num <= train_percent:
      return 0  # training collection
    if rand_num <= train_percent + eval_percent:
      return 1  # evaluate collection
    return 2  # test collection

  (train_data, evaluate_data, test_data) = (
      pipeline
      | 'Read Input Files'
      >> beam.io.ReadFromText(input_file_path, coder=coder_with_target)
      | 'Partition Data Source'
      >> beam.Partition(_partition_fn, 3))

  # TODO(b/32726166) Update input_format and format_metadata to read from these
  # values directly from the coder.
  (metadata, train_features, eval_features, test_features) = (
      (train_data, evaluate_data, test_data)
      | 'Preprocess'
      >> ml.Preprocess(feature_set, input_format='csv',
                       format_metadata={'headers': column_names}))

  # pylint: disable=expression-not-assigned
  (metadata | 'SaveMetadata'
   >> io.SaveMetadata(os.path.join(output_dir, 'metadata.json')))

  (train_features | 'SaveTrain'
                  >> io.SaveFeatures(
                      os.path.join(output_dir, 'features_train')))
  if eval_percent > 0:
    (eval_features | 'SaveEval'
                   >> io.SaveFeatures(
                       os.path.join(output_dir, 'features_eval')))
  if test_percent > 0:
    (test_features | 'SaveTest'
                      >> io.SaveFeatures(
                          os.path.join(output_dir, 'features_test')))
  # pylint: enable=expression-not-assigned


def run_dataflow(feature_set, column_names, input_file_path, train_percent,
                 eval_percent, test_percent, output_dir, cloud, project_id,
                 job_name):
  """Run Preprocessing as a Dataflow pipeline."""

  # Configure the pipeline.
  if cloud:
    options = {
        'staging_location': os.path.join(output_dir, 'tmp', 'staging'),
        'job_name': job_name,
        'project': project_id,
        'extra_packages': [ml.sdk_location],
        'teardown_policy': 'TEARDOWN_ALWAYS',
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    p = beam.Pipeline('DataflowPipelineRunner', options=opts)
  else:
    p = beam.Pipeline('DirectPipelineRunner')

  # Buid the pipeline.
  preprocess(
      pipeline=p,
      feature_set=feature_set,
      column_names=column_names,
      input_file_path=input_file_path,
      train_percent=train_percent,
      eval_percent=eval_percent,
      test_percent=test_percent,
      output_dir=output_dir)

  # Finally, run the pipeline.
  p.run()


def main(argv=None):
  args = parse_arguments(sys.argv if argv is None else argv)

  feature_set, column_names = preprocessing_features(args)

  run_dataflow(
      feature_set=feature_set,
      column_names=column_names,
      input_file_path=args.input_file_path,
      train_percent=args.train_percent,
      eval_percent=args.eval_percent,
      test_percent=args.test_percent,
      output_dir=args.output_dir,
      cloud=args.cloud,
      project_id=args.project_id,
      job_name=args.job_name)

if __name__ == '__main__':
  main()
