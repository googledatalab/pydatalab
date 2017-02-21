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

#from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.session_bundle import manifest_pb2
from tensorflow.python.lib.io import file_io


UNKNOWN_LABEL = 'ERROR_UNKNOWN_LABEL'
FEATURES_EXAMPLE_DICT_KEY = 'features_example_dict_key'
EXAMPLES_PLACEHOLDER_TENSOR_NAME = 'input_csv_string'

# Constants for the Prediction Graph fetch tensors.
PG_KEY = 'key_from_input'
PG_TARGET = 'target_from_input'

PG_REGRESSION_PREDICTED_TARGET = 'predicted_target'
PG_CLASSIFICATION_LABEL_TEMPLATE = 'top_%s_label'
PG_CLASSIFICATION_SCORE_TEMPLATE = 'top_%s_score'

# If input has the target label, we also give its score (which might not be in 
# the top n). 
# todo(brandondutra): get this working and use it.
PG_CLASSIFICATION_INPUT_TARGET_SCORE = 'score_of_input_target' 

# Constants for the exported input and output collections.
INPUT_COLLECTION_NAME = 'inputs'
OUTPUT_COLLECTION_NAME = 'outputs'

def get_placeholder_input_fn(train_config, preprocess_output_dir, model_type):
  """Input layer for the exported graph."""

  def get_input_features():
    """Read the input features from a placeholder example string tensor."""
    examples = tf.placeholder(
        dtype=tf.string,
        shape=(None,),
        name=EXAMPLES_PLACEHOLDER_TENSOR_NAME)

    # Parts is batch-size x num-columns sparse tensor. This means when running
    # prediction, all input rows should have a target column as the first
    # column, or all input rows should have the target column missing.
    # The condition below checks how many columns are in parts, and appends a 
    # ',' to the csv 'examples' placeholder string if a column is missing.
    parts = tf.string_split(examples, delimiter=',')
    new_examples = tf.cond(
        tf.less(tf.shape(parts)[1], len(train_config['csv_header'])),
        lambda: tf.string_join([tf.constant(','), tf.identity(examples)]),
        lambda: tf.identity(examples))
    features = util.parse_example_tensor(examples=new_examples,
                                         train_config=train_config)

    #global FEATURES_EXAMPLE_DICT_KEY
    #while FEATURES_EXAMPLE_DICT_KEY in features:
    #  FEATURES_EXAMPLE_DICT_KEY = '_' + FEATURES_EXAMPLE_DICT_KEY

    #features[FEATURES_EXAMPLE_DICT_KEY] = new_examples

    target = features.pop(train_config['target_column'])
    features, target = util.preprocess_input(
        features=features,
        target=target,
        train_config=train_config,
        preprocess_output_dir=preprocess_output_dir,
        model_type=model_type)
    # The target feature column is not used for prediction so return None.

    # Put target back in so it can be used when making the exported graph.
    features[train_config['target_column']] = target
    return features, None

  # Return a function to input the feaures into the model from a placeholder.
  return get_input_features


def get_reader_input_fn(train_config, preprocess_output_dir, model_type,
                        data_paths, batch_size, shuffle):
  """Builds input layer for training."""

  def get_input_features():
    """Read the input features from the given data paths."""
    _, examples = util.read_examples(data_paths, batch_size, shuffle)
    features = util.parse_example_tensor(examples=examples,
                                         train_config=train_config)

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




def get_export_signature_fn(train_config, args):
  """Builds the output layer in the exported graph.

  Also sets up the tensor names when calling session.run
  """

  def get_export_signature(examples, features, predictions):
    """Create an export signature with named input and output signatures."""
    target_name = train_config['target_column']
    key_name = train_config['key_column']

    if util.is_classification_model(args.model_type):
      
      # Get the label of the input target.
      string_value = util.get_vocabulary(args.preprocess_output_dir, target_name)
      input_target_label = tf.contrib.lookup.index_to_string(
          features[target_name],
          mapping=string_value,
          default_value='UNKNOWN')   

      outputs = {
          PG_KEY: tf.squeeze(features[key_name]).name,
          PG_TARGET: tf.squeeze(input_target_label).name,
      }

      # TODO(brandondutra): get the score of the target label too.
      #input_target_score = vector_slice(predictions, features[target_name])
      
      # get top k labels and their scores.
      (top_k_values, top_k_indices) = tf.nn.top_k(predictions, k=args.top_n)
      top_k_labels = tf.contrib.lookup.index_to_string(
          tf.to_int64(top_k_indices),
          mapping=string_value)
   
      # Write the top_k values using 2*top_k columns. 
      num_digits = int(math.ceil(math.log(args.top_n, 10)))
      if num_digits == 0:
        num_digits = 1
      for i in range(0, args.top_n):
        # Pad i based on the size of k. So if k = 100, i = 23 -> i = '023'. This
        # makes sorting the columns easy.
        padded_i = str(i+1).zfill(num_digits)

        label_alias = PG_CLASSIFICATION_LABEL_TEMPLATE % padded_i
        label_tensor_name = (tf.squeeze(
              tf.slice(top_k_labels, 
                       [0, i],
                       [tf.shape(top_k_labels)[0], 1])).name)
        score_alias = PG_CLASSIFICATION_SCORE_TEMPLATE % padded_i
        score_tensor_name = (tf.squeeze(
            tf.slice(top_k_values, 
                     [0, i], 
                     [tf.shape(top_k_values)[0], 1])).name)

        outputs.update({label_alias: label_tensor_name,
                        score_alias: score_tensor_name})

    else:
      outputs = {
          PG_KEY: tf.squeeze(features[key_name]).name,
          PG_TARGET: tf.squeeze(features[target_name]).name,
          PG_REGRESSION_PREDICTED_TARGET: tf.squeeze(predictions).name,
      }


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

    #input_placeholder_for_prediction = get_placeholder_input_fn(
    #    train_config, args.preprocess_output_dir, args.model_type)

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

    export_strategy_csv = util.make_export_strategy(
        train_config=train_config, 
        args=args, 
        input_type='csv',
        keep_target=True,
        assets_extra=additional_assets)
    export_strategy_json = util.make_export_strategy(
        train_config=train_config, 
        args=args, 
        input_type='json',
        keep_target=False,
        assets_extra=additional_assets)    

    input_reader_for_train = get_reader_input_fn(
        train_config, args.preprocess_output_dir, args.model_type,
        args.train_data_paths, args.batch_size, shuffle=True)
    input_reader_for_eval = get_reader_input_fn(
        train_config, args.preprocess_output_dir, args.model_type,
        args.eval_data_paths, args.eval_batch_size, shuffle=False)

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
        export_strategies=[export_strategy_csv, export_strategy_json],
        min_eval_frequency=args.min_eval_frequency,
        #eval_metrics=eval_metrics
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
  parser.add_argument('--learning_rate', type=float, default=0.01)
  parser.add_argument('--epsilon', type=float, default=0.0005)
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
