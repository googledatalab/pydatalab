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

from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.learn.python.learn import export_strategy
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import resources
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import saver
from tensorflow.python.util import compat


from . import feature_transforms
from . import feature_analysis

# Constants for the Prediction Graph fetch tensors.
PG_TARGET = 'target'  # from input

PG_REGRESSION_PREDICTED_TARGET = 'predicted'

PG_CLASSIFICATION_FIRST_LABEL = 'predicted'
PG_CLASSIFICATION_FIRST_SCORE = 'probability'
PG_CLASSIFICATION_LABEL_TEMPLATE = 'predicted_%s'
PG_CLASSIFICATION_SCORE_TEMPLATE = 'probability_%s'


class DatalabParser():
  """An arg parser that also prints package specific args with --datalab-help.

  When using Datalab magic's to run this trainer, it prints it's own help menu
  that describes the required options that are common to all trainers. In order
  to print just the options that are unique to this trainer, datalab calls this
  file with --datalab-help.

  This class implements --datalab-help by building a list of help string that only
  includes the unique parameters.
  """

  def __init__(self, epilog=None, datalab_epilog=None):
    self.full_parser = argparse.ArgumentParser(epilog=epilog)
    self.datalab_help = []
    self.datalab_epilog = datalab_epilog

    # Datalab help string
    self.full_parser.add_argument(
        '--datalab-help', action=self.make_datalab_help_action(),
        help='Show a smaller help message for DataLab only and exit')

    # The arguments added here are required to exist by Datalab's "%%ml train" magics.
    self.full_parser.add_argument(
        '--train', type=str, required=True, action='append', metavar='FILE')
    self.full_parser.add_argument(
        '--eval', type=str, required=True, action='append', metavar='FILE')
    self.full_parser.add_argument('--job-dir', type=str, required=True)
    self.full_parser.add_argument(
        '--analysis', type=str,
        metavar='ANALYSIS_OUTPUT_DIR',
        help=('Output folder of analysis. Should contain the schema, stats, and '
              'vocab files. Path must be on GCS if running cloud training. ' +
              'If absent, --schema and --features must be provided and ' +
              'the master trainer will do analysis locally.'))
    self.full_parser.add_argument(
        '--transform', action='store_true', default=False,
        help='If used, input data is raw csv that needs transformation. If analysis ' +
             'is required to run in trainerm this is automatically set to true.')
    self.full_parser.add_argument(
        '--schema', type=str,
        help='Schema of the training csv file. Only needed if analysis is required.')
    self.full_parser.add_argument(
        '--features', type=str,
        help='Feature transform config. Only needed if analysis is required.')

  def make_datalab_help_action(self):
    """Custom action for --datalab-help.

    The action output the package specific parameters and will be part of "%%ml train"
    help string.
    """
    datalab_help = self.datalab_help
    epilog = self.datalab_epilog

    class _CustomAction(argparse.Action):

      def __init__(self, option_strings, dest, help=None):
        super(_CustomAction, self).__init__(
            option_strings=option_strings, dest=dest, nargs=0, help=help)

      def __call__(self, parser, args, values, option_string=None):
        print('\n\n'.join(datalab_help))
        if epilog:
          print(epilog)

        # We have printed all help string datalab needs. If we don't quit, it will complain about
        # missing required arguments later.
        quit()
    return _CustomAction

  def add_argument(self, name, **kwargs):
    # Any argument added here is not required by Datalab, and so is unique
    # to this trainer. Add each argument to the main parser and the datalab helper string.
    self.full_parser.add_argument(name, **kwargs)
    name = name.replace('--', '')
    # leading spaces are needed for datalab's help formatting.
    msg = '  ' + name + ': '
    if 'help' in kwargs:
      msg += kwargs['help'] + ' '
    if kwargs.get('required', False):
      msg += 'Required. '
    else:
      msg += 'Optional. '
    if 'choices' in kwargs:
      msg += 'One of ' + str(kwargs['choices']) + '. '
    if 'default' in kwargs:
      msg += 'default: ' + str(kwargs['default']) + '.'
    self.datalab_help.append(msg)

  def parse_known_args(self, args=None):
    return self.full_parser.parse_known_args(args=args)


def parse_arguments(argv):
  """Parse the command line arguments."""
  parser = DatalabParser(
      epilog=('Note that if using a DNN model, --hidden-layer-size1=NUM, '
              '--hidden-layer-size2=NUM, ..., is also required. '),
      datalab_epilog=("""
  Note that if using a DNN model,
  hidden-layer-size1: NUM
  hidden-layer-size2: NUM
  ...
  is also required. """))

  # HP parameters
  parser.add_argument(
      '--epsilon', type=float, default=0.0005, metavar='R',
      help='tf.train.AdamOptimizer epsilon. Only used in dnn models.')
  parser.add_argument(
      '--l1-regularization', type=float, default=0.0, metavar='R',
      help='L1 term for linear models.')
  parser.add_argument(
      '--l2-regularization', type=float, default=0.0, metavar='R',
      help='L2 term for linear models.')

  # Model parameters
  parser.add_argument(
    '--model', required=True,
    choices=['linear_classification', 'linear_regression', 'dnn_classification', 'dnn_regression'])
  parser.add_argument(
      '--top-n', type=int, default=0, metavar='N',
      help=('For classification problems, the output graph will contain the '
            'labels and scores for the top n classes, and results will be in the form of '
            '"predicted, predicted_2, ..., probability, probability_2, ...". '
            'If --top-n=0, then all labels and scores are returned in the form of '
            '"predicted, class_name1, class_name2,...".'))

  # HP parameters
  parser.add_argument(
      '--learning-rate', type=float, default=0.01, metavar='R',
      help='optimizer learning rate.')

  # Training input parameters
  parser.add_argument(
      '--max-steps', type=int, metavar='N',
      help='Maximum number of training steps to perform. If unspecified, will '
           'honor "max-epochs".')
  parser.add_argument(
      '--max-epochs', type=int, default=1000, metavar='N',
      help='Maximum number of training data epochs on which to train. If '
           'both "max-steps" and "max-epochs" are specified, the training '
           'job will run for "max-steps" or "num-epochs", whichever occurs '
           'first. If early stopping is enabled, training may also stop '
           'earlier.')
  parser.add_argument(
      '--train-batch-size', type=int, default=64, metavar='N',
      help='How many training examples are used per step. If num-epochs is '
           'used, the last batch may not be full.')
  parser.add_argument(
      '--eval-batch-size', type=int, default=64, metavar='N',
      help='Batch size during evaluation. Larger values increase performance '
           'but also increase peak memory usgae on the master node. One pass '
           'over the full eval set is performed per evaluation run.')
  parser.add_argument(
      '--min-eval-frequency', type=int, default=1000, metavar='N',
      help='Minimum number of training steps between evaluations. Evaluation '
           'does not occur if no new checkpoint is available, hence, this is '
           'the minimum. If 0, the evaluation will only happen after training. ')
  parser.add_argument(
      '--early-stopping-num_evals', type=int, default=3,
      help='Automatic training stop after results of specified number of evals '
           'in a row show the model performance does not improve. Set to 0 to '
           'disable early stopping.')
  parser.add_argument(
      '--logging-level', choices=['error', 'warning', 'info'],
      help='The TF logging level. If absent, use info for cloud training '
           'and warning for local training.')

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
    elif (transform_name == feature_transforms.ONE_HOT_TRANSFORM or
          transform_name == feature_transforms.MULTI_HOT_TRANSFORM):
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
        new_feature = tf.contrib.layers.one_hot_column(sparse_ids)
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
          dimension=feature_transforms.IMAGE_HIDDEN_TENSOR_SIZE)
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

  if is_classification_model(args.model):

    # build maps from ints to the origional categorical strings.
    class_names = read_vocab(args, target_name)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        mapping=class_names,
        default_value='UNKNOWN')

    # Get the label of the input target.
    if keep_target:
      input_target_label = table.lookup(input_ops.features[target_name])
      outputs[PG_TARGET] = tf.squeeze(input_target_label)

    # TODO(brandondutra): get the score of the target label too.
    probabilities = model_fn_ops.predictions['probabilities']

    # if top_n == 0, this means use all the classes. We will use class names as
    # probabilities labels.
    if args.top_n == 0:
      predicted_index = tf.argmax(probabilities, axis=1)
      predicted = table.lookup(predicted_index)
      outputs.update({PG_CLASSIFICATION_FIRST_LABEL: predicted})
      probabilities_list = tf.unstack(probabilities, axis=1)
      for class_name, p in zip(class_names, probabilities_list):
        outputs[class_name] = p
    else:
      top_n = args.top_n

      # get top k labels and their scores.
      (top_k_values, top_k_indices) = tf.nn.top_k(probabilities, k=top_n)
      top_k_labels = table.lookup(tf.to_int64(top_k_indices))

      # Write the top_k values using 2*top_n columns.
      num_digits = int(math.ceil(math.log(top_n, 10)))
      if num_digits == 0:
        num_digits = 1
      for i in range(0, top_n):
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

      input_ops = feature_transforms.build_csv_serving_tensors_for_training_step(
          args.analysis, features, schema, stats, keep_target)
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

      if (model_fn_ops.scaffold is not None and
         model_fn_ops.scaffold.saver is not None):
        saver_for_restore = model_fn_ops.scaffold.saver
      else:
        saver_for_restore = saver.Saver(sharded=True)

      with tf_session.Session('') as session:
        saver_for_restore.restore(session, checkpoint_path)
        init_op = control_flow_ops.group(
            variables.local_variables_initializer(),
            resources.initialize_resources(resources.shared_resources()),
            tf.tables_initializer())

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


def get_estimator(args, output_dir, features, stats, target_vocab_size):
  # Check layers used for dnn models.
  if is_dnn_model(args.model) and not args.hidden_layer_sizes:
    raise ValueError('--hidden-layer-size* must be used with DNN models')
  if is_linear_model(args.model) and args.hidden_layer_sizes:
    raise ValueError('--hidden-layer-size* cannot be used with linear models')

  # Build tf.learn features
  feature_columns = build_feature_columns(features, stats, args.model)

  # Set how often to run checkpointing in terms of steps.
  config = tf.contrib.learn.RunConfig(
      save_checkpoints_steps=args.min_eval_frequency)

  train_dir = os.path.join(output_dir, 'train')
  if args.model == 'dnn_regression':
    estimator = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=args.hidden_layer_sizes,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.AdamOptimizer(
            args.learning_rate, epsilon=args.epsilon))
  elif args.model == 'linear_regression':
    estimator = tf.contrib.learn.LinearRegressor(
        feature_columns=feature_columns,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.FtrlOptimizer(
            args.learning_rate,
            l1_regularization_strength=args.l1_regularization,
            l2_regularization_strength=args.l2_regularization))
  elif args.model == 'dnn_classification':
    estimator = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=args.hidden_layer_sizes,
        n_classes=target_vocab_size,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.AdamOptimizer(
            args.learning_rate, epsilon=args.epsilon))
  elif args.model == 'linear_classification':
    estimator = tf.contrib.learn.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=target_vocab_size,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.FtrlOptimizer(
            args.learning_rate,
            l1_regularization_strength=args.l1_regularization,
            l2_regularization_strength=args.l2_regularization))
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
  vocab_path = os.path.join(args.analysis,
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
    schema_path_with_target = os.path.join(args.analysis,
                                           feature_transforms.SCHEMA_FILE)
    features_path = os.path.join(args.analysis,
                                 feature_transforms.FEATURES_FILE)
    stats_path = os.path.join(args.analysis,
                              feature_transforms.STATS_FILE)

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
    if args.transform:
      if any(v['transform'] == feature_transforms.IMAGE_TRANSFORM
             for k, v in six.iteritems(features)):
        raise ValueError('"image_to_vec" transform requires transformation step. ' +
                         'Cannot train from raw data.')

      input_reader_for_train = feature_transforms.build_csv_transforming_training_input_fn(
          schema=schema,
          features=features,
          stats=stats,
          analysis_output_dir=args.analysis,
          raw_data_file_pattern=args.train,
          training_batch_size=args.train_batch_size,
          num_epochs=args.max_epochs,
          randomize_input=True,
          min_after_dequeue=10,
          reader_num_threads=multiprocessing.cpu_count())
      input_reader_for_eval = feature_transforms.build_csv_transforming_training_input_fn(
          schema=schema,
          features=features,
          stats=stats,
          analysis_output_dir=args.analysis,
          raw_data_file_pattern=args.eval,
          training_batch_size=args.eval_batch_size,
          num_epochs=1,
          randomize_input=False,
          reader_num_threads=multiprocessing.cpu_count())
    else:
      input_reader_for_train = feature_transforms.build_tfexample_transfored_training_input_fn(
          schema=schema,
          features=features,
          analysis_output_dir=args.analysis,
          raw_data_file_pattern=args.train,
          training_batch_size=args.train_batch_size,
          num_epochs=args.max_epochs,
          randomize_input=True,
          min_after_dequeue=10,
          reader_num_threads=multiprocessing.cpu_count())
      input_reader_for_eval = feature_transforms.build_tfexample_transfored_training_input_fn(
          schema=schema,
          features=features,
          analysis_output_dir=args.analysis,
          raw_data_file_pattern=args.eval,
          training_batch_size=args.eval_batch_size,
          num_epochs=1,
          randomize_input=False,
          reader_num_threads=multiprocessing.cpu_count())

    if args.early_stopping_num_evals == 0:
      train_monitors = None
    else:
      if is_classification_model(args.model):
        early_stop_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=input_reader_for_eval,
            every_n_steps=args.min_eval_frequency,
            early_stopping_rounds=(args.early_stopping_num_evals * args.min_eval_frequency),
            early_stopping_metric='accuracy',
            early_stopping_metric_minimize=False)
      else:
        early_stop_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=input_reader_for_eval,
            every_n_steps=args.min_eval_frequency,
            early_stopping_rounds=(args.early_stopping_num_evals * args.min_eval_frequency))
      train_monitors = [early_stop_monitor]

    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=input_reader_for_train,
        eval_input_fn=input_reader_for_eval,
        train_steps=args.max_steps,
        train_monitors=train_monitors,
        export_strategies=[export_strategy_csv_notarget, export_strategy_csv_target],
        min_eval_frequency=args.min_eval_frequency,
        eval_steps=None)

  # Return a function to create an Experiment.
  return get_experiment


def local_analysis(args):
  if args.analysis:
    # Already analyzed.
    return

  if not args.schema or not args.features:
    raise ValueError('Either --analysis, or both --schema and --features are provided.')

  tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
  cluster_spec = tf_config.get('cluster', {})
  if len(cluster_spec.get('worker', [])) > 0:
    raise ValueError('If "schema" and "features" are provided, local analysis will run and ' +
                     'only BASIC scale-tier (no workers node) is supported.')

  if cluster_spec and not (args.schema.startswith('gs://') and args.features.startswith('gs://')):
    raise ValueError('Cloud trainer requires GCS paths for --schema and --features.')

  print('Running analysis.')
  schema = json.loads(file_io.read_file_to_string(args.schema).decode())
  features = json.loads(file_io.read_file_to_string(args.features).decode())
  args.analysis = os.path.join(args.job_dir, 'analysis')
  args.transform = True
  file_io.recursive_create_dir(args.analysis)
  feature_analysis.run_local_analysis(args.analysis, args.train, schema, features)
  print('Analysis done.')


def set_logging_level(args):
  if 'TF_CONFIG' in os.environ:
    tf.logging.set_verbosity(tf.logging.INFO)
  else:
    tf.logging.set_verbosity(tf.logging.ERROR)
  if args.logging_level == 'error':
    tf.logging.set_verbosity(tf.logging.ERROR)
  elif args.logging_level == 'warning':
    tf.logging.set_verbosity(tf.logging.WARN)
  elif args.logging_level == 'info':
    tf.logging.set_verbosity(tf.logging.INFO)


def main(argv=None):
  args = parse_arguments(sys.argv if argv is None else argv)
  local_analysis(args)
  set_logging_level(args)
  # Supress TensorFlow Debugging info.
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

  learn_runner.run(
      experiment_fn=get_experiment_fn(args),
      output_dir=args.job_dir)


if __name__ == '__main__':
  main()
