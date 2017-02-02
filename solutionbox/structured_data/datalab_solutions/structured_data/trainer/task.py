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

# TODO(brnaondutra): raise some excpetion vs print/sys.exit.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys

from . import util
import tensorflow as tf

import google.cloud.ml as ml

from tensorflow.contrib import metrics as metrics_lib
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.session_bundle import manifest_pb2

UNKNOWN_LABEL = 'ERROR_UNKNOWN_LABEL'
FEATURES_EXAMPLE_DICT_KEY = 'features_example_dict_key'
EXAMPLES_PLACEHOLDER_TENSOR_NAME = 'input_csv_string'

# Constants for prediction graph fetch tensors.
TARGET_SCORE_TENSOR_NAME = 'target_score_prediction'
TARGET_CLASS_TENSOR_NAME = 'target_class_prediction'
TARGET_INPUT_TENSOR_NAME = 'target_from_input'

# Constants for the exported input and output collections.
INPUT_COLLECTION_NAME = 'inputs'
OUTPUT_COLLECTION_NAME = 'outputs'



def get_placeholder_input_fn(train_config, preprocess_output_dir):
  """Input layer for the exported graph."""

  def get_input_features():
    """Read the input features from a placeholder example string tensor."""
    examples = tf.placeholder(
        dtype=tf.string,
        shape=(None,),
        name=EXAMPLES_PLACEHOLDER_TENSOR_NAME)

    features = util.parse_example_tensor(examples=examples,
                                         train_config=train_config)

    if FEATURES_EXAMPLE_DICT_KEY in features:
      print('ERROR: %s is a reserved feature name, please use a different'
            'feature name' % FEATURES_EXAMPLE_DICT_KEY)
      sys.exit(1)

    features[FEATURES_EXAMPLE_DICT_KEY] = examples

    target = features.pop(train_config['target_column'])
    features, target = util.preprocess_input(features, target, train_config, preprocess_output_dir)
    # The target feature column is not used for prediction so return None.
    return features, None

  # Return a function to input the feaures into the model from a placeholder.
  return get_input_features


def get_reader_input_fn(train_config, preprocess_output_dir, data_paths, batch_size,
                        shuffle):
  """Builds input layer for training."""

  def get_input_features():
    """Read the input features from the given data paths."""
    _, examples = util.read_examples(data_paths, batch_size, shuffle)
    features = util.parse_example_tensor(examples=examples,
                                         train_config=train_config)

    target_name = train_config['target_column']
    target = features.pop(target_name)
    features, target = util.preprocess_input(features, target, train_config, preprocess_output_dir)


    return features, target

  # Return a function to input the feaures into the model from a data path.
  return get_input_features


def get_export_signature_fn(train_config, args):
  """Builds the output layer in the exported graph.

  Also sets up the tensor names when calling session.run
  """

  def get_export_signature(examples, features, predictions):
    """Create an export signature with named input and output signatures."""
    target_name = train_config['target_column']
    key_name = train_config['key_column']
    outputs = {TARGET_SCORE_TENSOR_NAME: predictions.name,
               key_name: tf.squeeze(features[key_name]).name,
               #TARGET_INPUT_TENSOR_NAME: tf.squeeze(features[target_name]).name
               }

    predictions = tf.Print(predictions, [predictions])
    if util.is_classification_model(args.model_type):
      #_, string_value = util.get_vocabulary(args.preprocess_output_dir, target_name)
      #prediction = tf.argmax(predictions, 1)
      #labels = tf.contrib.lookup.index_to_string(
      #    prediction,
      #    mapping=string_value,
      #    default_value=train_config['csv_defaults'][target_name])
      #outputs.update({TARGET_CLASS_TENSOR_NAME: labels.name})
      pass

    inputs = {EXAMPLES_PLACEHOLDER_TENSOR_NAME: examples.name}

    tf.add_to_collection(INPUT_COLLECTION_NAME, json.dumps(inputs))
    tf.add_to_collection(OUTPUT_COLLECTION_NAME, json.dumps(outputs))

    input_signature = manifest_pb2.Signature()
    output_signature = manifest_pb2.Signature()

    for name, tensor_name in outputs.iteritems():
      output_signature.generic_signature.map[name].tensor_name = tensor_name

    for name, tensor_name in inputs.iteritems():
      input_signature.generic_signature.map[name].tensor_name = tensor_name

    # Return None for default classification signature.
    return None, {INPUT_COLLECTION_NAME: input_signature,
                  OUTPUT_COLLECTION_NAME: output_signature}

  # Return a function to create an export signature.
  return get_export_signature


def get_experiment_fn(args):
  """Builds the experiment function for learn_runner.run"""

  def get_experiment(output_dir):
    # Merge schema, input features, and transforms.
    train_config = util.merge_metadata(args.preprocess_output_dir, args.transforms_file)

    #metadata = ml.features.FeatureMetadata.get_metadata(args.metadata_path)
    #transform_config = json.loads(ml.util._file.load_file(args.transforms_file))
    #schema_config = json.loads(ml.util._file.load_file(args.schema_file))

    # Get the model to train.
    estimator = util.get_estimator(output_dir, train_config, args)

    input_placeholder_for_prediction = get_placeholder_input_fn(train_config, args.preprocess_output_dir)

    # Save the finished model to output_dir/model
    export_monitor = util.ExportLastModelMonitor(
        output_dir=output_dir,
        final_model_location='model',  # Relative to the output_dir.
        additional_assets=[args.transforms_file],
        input_fn=input_placeholder_for_prediction,
        input_feature_key=FEATURES_EXAMPLE_DICT_KEY,
        signature_fn=get_export_signature_fn(train_config, args))

    input_reader_for_train = get_reader_input_fn(
        train_config, args.preprocess_output_dir, args.train_data_paths, args.batch_size, shuffle=True)
    input_reader_for_eval = get_reader_input_fn(
        train_config, args.preprocess_output_dir, args.eval_data_paths, args.eval_batch_size, shuffle=False)

    # Set the eval metrics.
    # todo(brandondutra): make this work with HP tuning.
    if util.is_classification_model(args.model_type):
      streaming_accuracy = metrics_lib.streaming_accuracy
      eval_metrics =  {
            ('accuracy', 'classes'): streaming_accuracy,
            # Export the accuracy as a metric for hyperparameter tuning.
            ('training/hptuning/metric', 'classes'): streaming_accuracy
        }
    else:
      eval_metrics = None

    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=input_reader_for_train,
        eval_input_fn=input_reader_for_eval,
        train_steps=args.max_steps,
        train_monitors=[export_monitor],
        min_eval_frequency=args.min_eval_frequency,
        eval_metrics=eval_metrics)

  # Return a function to create an Experiment.
  return get_experiment


def parse_arguments(argv):
  """Parse the command line arguments."""
  parser = argparse.ArgumentParser()

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
  parser.add_argument('--learning_rate', type=float, default=0.01)
  parser.add_argument('--epsilon', type=float, default=0.0005)

  # Model problems
  parser.add_argument('--model_type',
                      choices=['linear_classification', 'linear_regression',
                               'dnn_classification', 'dnn_regression'],
                      required=True)
  # Training input parameters
  parser.add_argument('--layer_sizes', type=int, nargs='*')
  parser.add_argument('--max_steps', type=int, default=5000,
                      help='Maximum number of training steps to perform.')
  parser.add_argument('--batch_size', type=int, default=1000)
  parser.add_argument('--eval_batch_size', type=int, default=100)
  parser.add_argument('--min_eval_frequency', type=int, default=1000)

  # Training output parameters
  parser.add_argument('--save_checkpoints_secs', type=int, default=600,
                      help=('How often the model should be checkpointed/saved '
                            'in seconds'))
  parser.add_argument('--every_n_steps', type=int, default=10000,
                      help=('How often to export the checkpointed file in terms'
                            ' of training steps. Should be large enough so that'
                            ' a new checkpoined model is saved before running '
                            'again.'))
  return parser.parse_args(args=argv[1:])


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
