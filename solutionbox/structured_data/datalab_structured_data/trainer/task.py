# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
import re
import sys
import math

from . import util
import tensorflow as tf
from tensorflow.contrib import metrics as metrics_lib

from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.lib.io import file_io


def get_reader_input_fn(train_config, preprocess_output_dir, model_type,
                        data_paths, batch_size, shuffle, num_epochs=None):
  """Builds input layer for training."""

  def get_input_features():
    """Read the input features from the given data paths."""
    _, examples = util.read_examples(
        input_files=data_paths,
        batch_size=batch_size,
        shuffle=shuffle,
        num_epochs=num_epochs)
    features = util.parse_example_tensor(examples=examples,
                                         train_config=train_config,
                                         keep_target=True)

    target_name = train_config['target_column']
    target = features.pop(target_name)
    features, target = util.preprocess_input(
        features=features,
        target=target,
        train_config=train_config,
        preprocess_output_dir=preprocess_output_dir,
        model_type=model_type)

    return features, target

  # Return a function to input the feaures into the model from a data path.
  return get_input_features


def get_experiment_fn(args):
  """Builds the experiment function for learn_runner.run.

  Args:
    args: the command line args

  Returns:
    A function that returns a tf.learn experiment object.
  """

  def get_experiment(output_dir):
    # Merge schema, input features, and transforms.
    train_config = util.merge_metadata(args.preprocess_output_dir,
                                       args.transforms_file)

    # Get the model to train.
    estimator = util.get_estimator(output_dir, train_config, args)

    # Save a copy of the scehma and input to the model folder.
    schema_file = os.path.join(args.preprocess_output_dir, util.SCHEMA_FILE)

    # Make list of files to save with the trained model.
    additional_assets = {'transforms.json': args.transforms_file,
                         util.SCHEMA_FILE: schema_file}
    if util.is_classification_model(args.model_type):
      target_name = train_config['target_column']
      vocab_file_name = util.CATEGORICAL_ANALYSIS % target_name
      vocab_file_path = os.path.join(
          args.preprocess_output_dir, vocab_file_name)
      assert file_io.file_exists(vocab_file_path)
      additional_assets[vocab_file_name] = vocab_file_path

    export_strategy_target = util.make_export_strategy(
        train_config=train_config,
        args=args,
        keep_target=True,
        assets_extra=additional_assets)
    export_strategy_notarget = util.make_export_strategy(
        train_config=train_config,
        args=args,
        keep_target=False,
        assets_extra=additional_assets)

    input_reader_for_train = get_reader_input_fn(
        train_config=train_config,
        preprocess_output_dir=args.preprocess_output_dir,
        model_type=args.model_type,
        data_paths=args.train_data_paths,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_epochs=args.num_epochs)

    input_reader_for_eval = get_reader_input_fn(
        train_config=train_config,
        preprocess_output_dir=args.preprocess_output_dir,
        model_type=args.model_type,
        data_paths=args.eval_data_paths,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_epochs=1)

    # Set the eval metrics.
    # TODO(brandondutra): make this work with HP tuning.
    if util.is_classification_model(args.model_type):
      streaming_accuracy = metrics_lib.streaming_accuracy
      eval_metrics = {
          ('accuracy', 'classes'): streaming_accuracy,
          # Export the accuracy as a metric for hyperparameter tuning.
          #('training/hptuning/metric', 'classes'): streaming_accuracy
      }
    else:
      eval_metrics = None

    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=input_reader_for_train,
        eval_input_fn=input_reader_for_eval,
        train_steps=args.max_steps,
        export_strategies=[export_strategy_target, export_strategy_notarget],
        min_eval_frequency=args.min_eval_frequency,
        eval_steps=None,
        )

  # Return a function to create an Experiment.
  return get_experiment


def parse_arguments(argv):
  """Parse the command line arguments."""
  parser = argparse.ArgumentParser(
      description=('Train a regression or classification model. Note that if '
                   'using a DNN model, --layer_size1=NUM, --layer_size2=NUM, '
                   'should be used. '))

  # I/O file parameters
  parser.add_argument('--train_data_paths', type=str, action='append',
                      required=True)
  parser.add_argument('--eval_data_paths', type=str, action='append',
                      required=True)
  parser.add_argument('--output_path', type=str, required=True)
  parser.add_argument('--preprocess_output_dir',
                      type=str,
                      required=True,
                      help=('Output folder of preprocessing. Should contain the'
                            ' files input_features.json, schema.json, and the'
                            ' optional files numerical_analysis.json and'
                            ' vocab_str1.csv. Path must be on GCS if running'
                            ' cloud training.'))
  parser.add_argument('--transforms_file',
                      type=str,
                      required=True,
                      help=('File describing the the transforms to apply on '
                            'each column'))

  # HP parameters
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='tf.train.AdamOptimizer learning rate')
  parser.add_argument('--epsilon', type=float, default=0.0005,
                      help='tf.train.AdamOptimizer epsilon')
  # --layer_size See below

  # Model problems
  parser.add_argument('--model_type',
                      choices=['linear_classification', 'linear_regression',
                               'dnn_classification', 'dnn_regression'],
                      required=True)
  parser.add_argument('--top_n',
                      type=int,
                      default=1,
                      help=('For classification problems, the output graph '
                            'will contain the labels and scores for the top '
                            'n classes.'))
  # Training input parameters
  parser.add_argument('--max_steps', type=int, default=5000,
                      help='Maximum number of training steps to perform.')
  parser.add_argument('--num_epochs',
                      type=int,
                      help=('Maximum number of training data epochs on which '
                            'to train. If both --max-steps and --num-epochs '
                            'are specified, the training job will run for '
                            '--max-steps or --num-epochs, whichever occurs '
                            'first. If unspecified will run for --max-steps.'))
  parser.add_argument('--train_batch_size', type=int, default=1000)
  parser.add_argument('--eval_batch_size', type=int, default=1000)
  parser.add_argument('--min_eval_frequency', type=int, default=100,
                      help=('Minimum number of training steps between '
                            'evaluations'))

  # Training output parameters
  parser.add_argument('--save_checkpoints_secs', type=int, default=600,
                      help=('How often the model should be checkpointed/saved '
                            'in seconds'))

  args, remaining_args = parser.parse_known_args(args=argv[1:])

  # All HP parambeters must be unique, so we need to support an unknown number
  # of --layer_size1=10 --layer_size2=10 ...
  # Look at remaining_args for layer_size\d+ to get the layer info.

  # Get number of layers
  pattern = re.compile('layer_size(\d+)')
  num_layers = 0
  for other_arg in remaining_args:
    match = re.search(pattern, other_arg)
    if match:
      num_layers = max(num_layers, int(match.group(1)))

  # Build a new parser so we catch unknown args and missing layer_sizes.
  parser = argparse.ArgumentParser()
  for i in range(num_layers):
    parser.add_argument('--layer_size%s' % str(i+1), type=int, required=True)

  layer_args = vars(parser.parse_args(args=remaining_args))
  layer_sizes = []
  for i in range(num_layers):
    key = 'layer_size%s' % str(i+1)
    layer_sizes.append(layer_args[key])

  assert len(layer_sizes) == num_layers
  args.layer_sizes = layer_sizes

  return args


def main(argv=None):
  """Run a Tensorflow model on the Iris dataset."""
  args = parse_arguments(sys.argv if argv is None else argv)

  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  # First find out if there's a task value on the environment variable.
  # If there is none or it is empty define a default one.
  task_data = env.get('task') or {'type': 'master', 'index': 0}

  trial = task_data.get('trial')
  if trial is not None:
    output_dir = os.path.join(args.output_path, trial)
  else:
    output_dir = args.output_path

  learn_runner.run(
      experiment_fn=get_experiment_fn(args),
      output_dir=output_dir)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main()
