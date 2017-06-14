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
import math
import multiprocessing
import os
import re
import sys
import six
import tensorflow as tf

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.learn.python.learn import export_strategy
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import saver
from tensorflow.python.util import compat


from . import feature_transforms

# Constants for the Prediction Graph fetch tensors.
PG_TARGET = 'target'  # from input

PG_REGRESSION_PREDICTED_TARGET = 'predicted'

PG_CLASSIFICATION_FIRST_LABEL = 'predicted'
PG_CLASSIFICATION_FIRST_SCORE = 'score'
PG_CLASSIFICATION_LABEL_TEMPLATE = 'predicted_%s'
PG_CLASSIFICATION_SCORE_TEMPLATE = 'score_%s'


def parse_arguments(argv):
  """Parse the command line arguments."""
  parser = argparse.ArgumentParser(
      description=('Train a regression or classification model. Note that if '
                   'using a DNN model, --layer-size1=NUM, --layer-size2=NUM, '
                   'should be used. '))

  # I/O file parameters
  parser.add_argument('--train-data-paths', type=str,
                      required=True, action='append')
  parser.add_argument('--eval-data-paths', type=str,
                      required=True, action='append')
  parser.add_argument('--job-dir', type=str, required=True)
  parser.add_argument('--analysis-output-dir',
                      type=str,
                      required=True,
                      help=('Output folder of analysis. Should contain the'
                            ' schema, stats, and vocab files.'
                            ' Path must be on GCS if running'
                            ' cloud training.'))
  parser.add_argument('--run-transforms',
                      action='store_true',
                      default=False,
                      help=('If used, input data is raw csv that needs '
                            'transformation.'))

  # HP parameters
  parser.add_argument('--learning-rate', type=float, default=0.01,
                      help='optimizer learning rate')
  parser.add_argument('--epsilon', type=float, default=0.0005,
                      help='tf.train.AdamOptimizer epsilon. Only used in '
                           'dnn models.')
  parser.add_argument('--l1_regularization', type=float, default=0.0,
                      help='L1 term for linear models.')
  parser.add_argument('--l2_regularization', type=float, default=0.0,
                      help='L2 term for linear models.')

  # Model problems
  parser.add_argument('--model-type',
                      choices=['linear_classification', 'linear_regression',
                               'dnn_classification', 'dnn_regression'],
                      required=True)
  parser.add_argument('--top-n',
                      type=int,
                      default=1,
                      help=('For classification problems, the output graph '
                            'will contain the labels and scores for the top '
                            'n classes.'))
  # Training input parameters
  parser.add_argument('--max-steps', type=int, default=5000,
                      help='Maximum number of training steps to perform.')
  parser.add_argument('--num-epochs',
                      type=int,
                      help=('Maximum number of training data epochs on which '
                            'to train. If both --max-steps and --num-epochs '
                            'are specified, the training job will run for '
                            '--max-steps or --num-epochs, whichever occurs '
                            'first. If unspecified will run for --max-steps.'))
  parser.add_argument('--train-batch-size', type=int, default=1000)
  parser.add_argument('--eval-batch-size', type=int, default=1000)
  parser.add_argument('--min-eval-frequency', type=int, default=100,
                      help=('Minimum number of training steps between '
                            'evaluations'))

  # other parameters
  parser.add_argument('--save-checkpoints-secs', type=int, default=600,
                      help=('How often the model should be checkpointed/saved '
                            'in seconds'))

  args, remaining_args = parser.parse_known_args(args=argv[1:])

  # All HP parambeters must be unique, so we need to support an unknown number
  # of --hidden-layer-size1=10 --lhidden-layer-size2=10 ...
  # Look at remaining_args for hidden-layer-size\d+ to get the layer info.

  # Get number of layers
  pattern = re.compile('hidden-layer-size(\d+)')
  num_layers = 0
  for other_arg in remaining_args:
    match = re.search(pattern, other_arg)
    if match:
      if int(match.group(1)) <= 0:
        raise ValueError('layer size must be a positive integer. Was given %s' % other_arg)
      num_layers = max(num_layers, int(match.group(1)))

  # Build a new parser so we catch unknown args and missing layer_sizes.
  parser = argparse.ArgumentParser()
  for i in range(num_layers):
    parser.add_argument('--hidden-layer-size%s' % str(i + 1), type=int, required=True)

  layer_args = vars(parser.parse_args(args=remaining_args))
  hidden_layer_sizes = []
  for i in range(num_layers):
    key = 'hidden_layer_size%s' % str(i + 1)
    hidden_layer_sizes.append(layer_args[key])

  assert len(hidden_layer_sizes) == num_layers
  args.hidden_layer_sizes = hidden_layer_sizes

  return args


def is_linear_model(model_type):
  return model_type.startswith('linear_')


def is_dnn_model(model_type):
  return model_type.startswith('dnn_')


def is_regression_model(model_type):
  return model_type.endswith('_regression')


def is_classification_model(model_type):
  return model_type.endswith('_classification')


def build_feature_columns(features, stats, model_type):
  feature_columns = []
  is_dnn = is_dnn_model(model_type)

  # Supported transforms:
  # for DNN
  #   numerical number
  #   one hot: sparse int column -> one_hot_column
  #   ebmedding: sparse int column -> embedding_column
  #   text: sparse int weighted column -> embedding_column
  # for linear
  #   numerical number
  #   one hot: sparse int column
  #   ebmedding: sparse int column -> hash int
  #   text: sparse int weighted column
  # It is unfortunate that tf.layers has different feature transforms if the
  # model is linear or DNN. This pacakge should not expose to the user that
  # we are using tf.layers.
  for name, transform in six.iteritems(features):
    transform_name = transform['transform']
    source_column = transform['source_column']

    if transform_name in feature_transforms.NUMERIC_TRANSFORMS:
      new_feature = tf.contrib.layers.real_valued_column(name, dimension=1)
    elif transform_name == feature_transforms.ONE_HOT_TRANSFORM:
      sparse = tf.contrib.layers.sparse_column_with_integerized_feature(
          name,
          bucket_size=stats['column_stats'][source_column]['vocab_size'])
      if is_dnn:
        new_feature = tf.contrib.layers.one_hot_column(sparse)
      else:
        new_feature = sparse
    elif transform_name == feature_transforms.EMBEDDING_TRANSFROM:
      if is_dnn:
        sparse = tf.contrib.layers.sparse_column_with_integerized_feature(
            name,
            bucket_size=stats['column_stats'][source_column]['vocab_size'])
        new_feature = tf.contrib.layers.embedding_column(
            sparse,
            dimension=transform['embedding_dim'])
      else:
        new_feature = tf.contrib.layers.sparse_column_with_hash_bucket(
            name,
            hash_bucket_size=transform['embedding_dim'],
            dtype=dtypes.int64)
    elif transform_name in feature_transforms.TEXT_TRANSFORMS:
      sparse_ids = tf.contrib.layers.sparse_column_with_integerized_feature(
          name + '_ids',
          bucket_size=stats['column_stats'][source_column]['vocab_size'],
          combiner='sum')
      sparse_weights = tf.contrib.layers.weighted_sparse_column(
          sparse_id_column=sparse_ids,
          weight_column_name=name + '_weights',
          dtype=dtypes.float32)
      if is_dnn:
        # TODO(brandondutra): Figure out why one_hot_column does not work.
        # Use new_feature = tf.contrib.layers.one_hot_column(sparse_weights)
        dimension = int(math.log(stats['column_stats'][source_column]['vocab_size'])) + 1
        new_feature = tf.contrib.layers.embedding_column(
            sparse_weights,
            dimension=dimension,
            combiner='sqrtn')
      else:
        new_feature = sparse_weights
    elif (transform_name == feature_transforms.TARGET_TRANSFORM or
          transform_name == feature_transforms.KEY_TRANSFORM):
      continue
    elif transform_name == feature_transforms.IMAGE_TRANSFORM:
      new_feature = tf.contrib.layers.real_valued_column(
          name,
          dimension=feature_transforms.IMAGE_BOTTLENECK_TENSOR_SIZE)
    else:
      raise ValueError('Unknown transfrom %s' % transform_name)

    feature_columns.append(new_feature)

  return feature_columns


def recursive_copy(src_dir, dest_dir):
  """Copy the contents of src_dir into the folder dest_dir.
  Args:
    src_dir: gsc or local path.
    dest_dir: gcs or local path.
  """

  file_io.recursive_create_dir(dest_dir)
  for file_name in file_io.list_directory(src_dir):
    old_path = os.path.join(src_dir, file_name)
    new_path = os.path.join(dest_dir, file_name)

    if file_io.is_directory(old_path):
      recursive_copy(old_path, new_path)
    else:
      file_io.copy(old_path, new_path, overwrite=True)


def make_prediction_output_tensors(args, features, input_ops, model_fn_ops,
                                   keep_target):
  """Makes the final prediction output layer."""
  target_name = feature_transforms.get_target_name(features)
  key_names = get_key_names(features)

  outputs = {}
  outputs.update({key_name: tf.squeeze(input_ops.features[key_name])
                  for key_name in key_names})

  if is_classification_model(args.model_type):

    # build maps from ints to the origional categorical strings.
    string_values = read_vocab(args, target_name)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        mapping=string_values,
        default_value='UNKNOWN')

    # Get the label of the input target.
    if keep_target:
      input_target_label = table.lookup(input_ops.features[target_name])
      outputs[PG_TARGET] = tf.squeeze(input_target_label)

    # TODO(brandondutra): get the score of the target label too.
    probabilities = model_fn_ops.predictions['probabilities']

    # get top k labels and their scores.
    (top_k_values, top_k_indices) = tf.nn.top_k(probabilities, k=args.top_n)
    top_k_labels = table.lookup(tf.to_int64(top_k_indices))

    # Write the top_k values using 2*top_k columns.
    num_digits = int(math.ceil(math.log(args.top_n, 10)))
    if num_digits == 0:
      num_digits = 1
    for i in range(0, args.top_n):
      # Pad i based on the size of k. So if k = 100, i = 23 -> i = '023'. This
      # makes sorting the columns easy.
      padded_i = str(i + 1).zfill(num_digits)

      if i == 0:
        label_alias = PG_CLASSIFICATION_FIRST_LABEL
      else:
        label_alias = PG_CLASSIFICATION_LABEL_TEMPLATE % padded_i

      label_tensor_name = (tf.squeeze(
          tf.slice(top_k_labels, [0, i], [tf.shape(top_k_labels)[0], 1])))

      if i == 0:
        score_alias = PG_CLASSIFICATION_FIRST_SCORE
      else:
        score_alias = PG_CLASSIFICATION_SCORE_TEMPLATE % padded_i

      score_tensor_name = (tf.squeeze(
          tf.slice(top_k_values,
                   [0, i],
                   [tf.shape(top_k_values)[0], 1])))

      outputs.update({label_alias: label_tensor_name,
                      score_alias: score_tensor_name})

  else:
    if keep_target:
      outputs[PG_TARGET] = tf.squeeze(input_ops.features[target_name])

    scores = model_fn_ops.predictions['scores']
    outputs[PG_REGRESSION_PREDICTED_TARGET] = tf.squeeze(scores)

  return outputs


# This function is strongly based on
# tensorflow/contrib/learn/python/learn/estimators/estimator.py:export_savedmodel()
# The difference is we need to modify estimator's output layer.
def make_export_strategy(
        args,
        keep_target,
        assets_extra,
        features,
        schema,
        stats):
  """Makes prediction graph that takes json input.

  Args:
    args: command line args
    keep_target: If ture, target column is returned in prediction graph. Target
        column must also exist in input data
    assets_extra: other fiels to copy to the output folder
    job_dir: root job folder
    features: features dict
    schema: schema list
    stats: stats dict
  """
  target_name = feature_transforms.get_target_name(features)
  csv_header = [col['name'] for col in schema]
  if not keep_target:
    csv_header.remove(target_name)

  def export_fn(estimator, export_dir_base, checkpoint_path=None, eval_result=None):
    with ops.Graph().as_default() as g:
      contrib_variables.create_global_step(g)

      input_ops = feature_transforms.build_csv_serving_tensors(
          args.analysis_output_dir, features, schema, stats, keep_target)
      model_fn_ops = estimator._call_model_fn(input_ops.features,
                                              None,
                                              model_fn_lib.ModeKeys.INFER)
      output_fetch_tensors = make_prediction_output_tensors(
          args=args,
          features=features,
          input_ops=input_ops,
          model_fn_ops=model_fn_ops,
          keep_target=keep_target)

      # Don't use signature_def_utils.predict_signature_def as that renames
      # tensor names if there is only 1 input/output tensor!
      signature_inputs = {key: tf.saved_model.utils.build_tensor_info(tensor)
                          for key, tensor in six.iteritems(input_ops.default_inputs)}
      signature_outputs = {key: tf.saved_model.utils.build_tensor_info(tensor)
                           for key, tensor in six.iteritems(output_fetch_tensors)}
      signature_def_map = {
          'serving_default':
              signature_def_utils.build_signature_def(
                  signature_inputs,
                  signature_outputs,
                  tf.saved_model.signature_constants.PREDICT_METHOD_NAME)}

      if not checkpoint_path:
        # Locate the latest checkpoint
        checkpoint_path = saver.latest_checkpoint(estimator._model_dir)
      if not checkpoint_path:
        raise ValueError("Couldn't find trained model at %s."
                         % estimator._model_dir)

      export_dir = saved_model_export_utils.get_timestamped_export_dir(
          export_dir_base)

      with tf_session.Session('') as session:
        variables.local_variables_initializer()
        data_flow_ops.tables_initializer()
        saver_for_restore = saver.Saver(
            variables.global_variables(),
            sharded=True)
        saver_for_restore.restore(session, checkpoint_path)

        init_op = control_flow_ops.group(
            variables.local_variables_initializer(),
            data_flow_ops.tables_initializer())

        # Perform the export
        builder = saved_model_builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(
            session, [tag_constants.SERVING],
            signature_def_map=signature_def_map,
            assets_collection=ops.get_collection(
                ops.GraphKeys.ASSET_FILEPATHS),
            legacy_init_op=init_op)
        builder.save(False)

      # Add the extra assets
      if assets_extra:
        assets_extra_path = os.path.join(compat.as_bytes(export_dir),
                                         compat.as_bytes('assets.extra'))
        for dest_relative, source in assets_extra.items():
          dest_absolute = os.path.join(compat.as_bytes(assets_extra_path),
                                       compat.as_bytes(dest_relative))
          dest_path = os.path.dirname(dest_absolute)
          file_io.recursive_create_dir(dest_path)
          file_io.copy(source, dest_absolute)

    # only keep the last 3 models
    saved_model_export_utils.garbage_collect_exports(
        export_dir_base,
        exports_to_keep=3)

    # save the last model to the model folder.
    # export_dir_base = A/B/intermediate_models/
    if keep_target:
      final_dir = os.path.join(args.job_dir, 'evaluation_model')
    else:
      final_dir = os.path.join(args.job_dir, 'model')
    if file_io.is_directory(final_dir):
      file_io.delete_recursively(final_dir)
    file_io.recursive_create_dir(final_dir)
    recursive_copy(export_dir, final_dir)

    return export_dir

  if keep_target:
    intermediate_dir = 'intermediate_evaluation_models'
  else:
    intermediate_dir = 'intermediate_prediction_models'

  return export_strategy.ExportStrategy(intermediate_dir, export_fn)


def make_feature_engineering_fn(features):
  """feature_engineering_fn for adding a hidden layer on image embeddings."""

  def feature_engineering_fn(feature_tensors_dict, target):
    engineered_features = {}
    for name, feature_tensor in six.iteritems(feature_tensors_dict):
      if name in features and features[name]['transform'] == feature_transforms.IMAGE_TRANSFORM:
        bottleneck_with_no_gradient = tf.stop_gradient(feature_tensor)
        with tf.name_scope(name, 'Wx_plus_b'):
          hidden = layers.fully_connected(
              bottleneck_with_no_gradient,
              int(feature_transforms.IMAGE_BOTTLENECK_TENSOR_SIZE / 4))
          engineered_features[name] = hidden
      else:
        engineered_features[name] = feature_tensor
    return engineered_features, target

  return feature_engineering_fn


def get_estimator(args, output_dir, features, stats, target_vocab_size):
  # Check layers used for dnn models.
  if is_dnn_model(args.model_type) and not args.hidden_layer_sizes:
    raise ValueError('--hidden-layer-size* must be used with DNN models')
  if is_linear_model(args.model_type) and args.hidden_layer_sizes:
    raise ValueError('--hidden-layer-size* cannot be used with linear models')

  # Build tf.learn features
  feature_columns = build_feature_columns(features, stats, args.model_type)
  feature_engineering_fn = make_feature_engineering_fn(features)

  # Set how often to run checkpointing in terms of time.
  config = tf.contrib.learn.RunConfig(
      save_checkpoints_secs=args.save_checkpoints_secs)

  train_dir = os.path.join(output_dir, 'train')
  if args.model_type == 'dnn_regression':
    estimator = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=args.hidden_layer_sizes,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.AdamOptimizer(
            args.learning_rate, epsilon=args.epsilon),
        feature_engineering_fn=feature_engineering_fn)
  elif args.model_type == 'linear_regression':
    estimator = tf.contrib.learn.LinearRegressor(
        feature_columns=feature_columns,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.FtrlOptimizer(
            args.learning_rate,
            l1_regularization_strength=args.l1_regularization,
            l2_regularization_strength=args.l2_regularization),
        feature_engineering_fn=feature_engineering_fn)
  elif args.model_type == 'dnn_classification':
    estimator = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=args.hidden_layer_sizes,
        n_classes=target_vocab_size,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.AdamOptimizer(
            args.learning_rate, epsilon=args.epsilon),
        feature_engineering_fn=feature_engineering_fn)
  elif args.model_type == 'linear_classification':
    estimator = tf.contrib.learn.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=target_vocab_size,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.FtrlOptimizer(
            args.learning_rate,
            l1_regularization_strength=args.l1_regularization,
            l2_regularization_strength=args.l2_regularization),
        feature_engineering_fn=feature_engineering_fn)
  else:
    raise ValueError('bad --model-type value')

  return estimator


def read_vocab(args, column_name):
  """Reads a vocab file if it exists.

  Args:
    args: command line flags
    column_name: name of column to that has a vocab file.

  Returns:
    List of vocab words or [] if the vocab file is not found.
  """
  vocab_path = os.path.join(args.analysis_output_dir,
                            feature_transforms.VOCAB_ANALYSIS_FILE % column_name)

  if not file_io.file_exists(vocab_path):
    return []

  vocab, _ = feature_transforms.read_vocab_file(vocab_path)
  return vocab


def get_key_names(features):
  names = []
  for name, transform in six.iteritems(features):
    if transform['transform'] == feature_transforms.KEY_TRANSFORM:
      names.append(name)
  return names


def read_json_file(file_path):
  if not file_io.file_exists(file_path):
    raise ValueError('File not found: %s' % file_path)
  return json.loads(file_io.read_file_to_string(file_path).decode())


def get_experiment_fn(args):
  """Builds the experiment function for learn_runner.run.

  Args:
    args: the command line args

  Returns:
    A function that returns a tf.learn experiment object.
  """

  def get_experiment(output_dir):
    # Read schema, input features, and transforms.
    schema_path_with_target = os.path.join(args.analysis_output_dir, feature_transforms.SCHEMA_FILE)
    features_path = os.path.join(args.analysis_output_dir, feature_transforms.FEATURES_FILE)
    stats_path = os.path.join(args.analysis_output_dir, feature_transforms.STATS_FILE)

    schema = read_json_file(schema_path_with_target)
    features = read_json_file(features_path)
    stats = read_json_file(stats_path)

    target_column_name = feature_transforms.get_target_name(features)
    if not target_column_name:
      raise ValueError('target missing from features file.')

    # Make a copy of the schema file without the target column.
    schema_without_target = [col for col in schema if col['name'] != target_column_name]
    schema_path_without_target = os.path.join(args.job_dir, 'schema_without_target.json')
    file_io.recursive_create_dir(args.job_dir)
    file_io.write_string_to_file(schema_path_without_target,
                                 json.dumps(schema_without_target, indent=2))

    # Make list of files to save with the trained model.
    additional_assets_with_target = {
        feature_transforms.FEATURES_FILE: features_path,
        feature_transforms.SCHEMA_FILE: schema_path_with_target}
    additional_assets_without_target = {
        feature_transforms.FEATURES_FILE: features_path,
        feature_transforms.SCHEMA_FILE: schema_path_without_target}

    # Get the model to train.
    target_vocab = read_vocab(args, target_column_name)
    estimator = get_estimator(args, output_dir, features, stats, len(target_vocab))

    export_strategy_csv_notarget = make_export_strategy(
        args=args,
        keep_target=False,
        assets_extra=additional_assets_without_target,
        features=features,
        schema=schema,
        stats=stats)
    export_strategy_csv_target = make_export_strategy(
        args=args,
        keep_target=True,
        assets_extra=additional_assets_with_target,
        features=features,
        schema=schema,
        stats=stats)

    # Build readers for training.
    if args.run_transforms:
      if any(v['transform'] == feature_transforms.IMAGE_TRANSFORM
             for k, v in six.iteritems(features)):
        raise ValueError('"image_to_vec" transform requires transformation step. ' +
                         'Cannot train from raw data.')

      input_reader_for_train = feature_transforms.build_csv_transforming_training_input_fn(
          schema=schema,
          features=features,
          stats=stats,
          analysis_output_dir=args.analysis_output_dir,
          raw_data_file_pattern=args.train_data_paths,
          training_batch_size=args.train_batch_size,
          num_epochs=args.num_epochs,
          randomize_input=True,
          min_after_dequeue=10,
          reader_num_threads=multiprocessing.cpu_count())
      input_reader_for_eval = feature_transforms.build_csv_transforming_training_input_fn(
          schema=schema,
          features=features,
          stats=stats,
          analysis_output_dir=args.analysis_output_dir,
          raw_data_file_pattern=args.eval_data_paths,
          training_batch_size=args.eval_batch_size,
          num_epochs=1,
          randomize_input=False,
          reader_num_threads=multiprocessing.cpu_count())
    else:
      input_reader_for_train = feature_transforms.build_tfexample_transfored_training_input_fn(
          schema=schema,
          features=features,
          analysis_output_dir=args.analysis_output_dir,
          raw_data_file_pattern=args.train_data_paths,
          training_batch_size=args.train_batch_size,
          num_epochs=args.num_epochs,
          randomize_input=True,
          min_after_dequeue=10,
          reader_num_threads=multiprocessing.cpu_count())
      input_reader_for_eval = feature_transforms.build_tfexample_transfored_training_input_fn(
          schema=schema,
          features=features,
          analysis_output_dir=args.analysis_output_dir,
          raw_data_file_pattern=args.eval_data_paths,
          training_batch_size=args.eval_batch_size,
          num_epochs=1,
          randomize_input=False,
          reader_num_threads=multiprocessing.cpu_count())

    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=input_reader_for_train,
        eval_input_fn=input_reader_for_eval,
        train_steps=args.max_steps,
        export_strategies=[export_strategy_csv_notarget, export_strategy_csv_target],
        min_eval_frequency=args.min_eval_frequency,
        eval_steps=None)

  # Return a function to create an Experiment.
  return get_experiment


def main(argv=None):
  args = parse_arguments(sys.argv if argv is None else argv)

  tf.logging.set_verbosity(tf.logging.INFO)
  learn_runner.run(
      experiment_fn=get_experiment_fn(args),
      output_dir=args.job_dir)


if __name__ == '__main__':
  main()
