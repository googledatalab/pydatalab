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
# ==============================================================================

import json
import multiprocessing
import os
import math
import six

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.contrib.learn.python.learn import export_strategy
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)

from tensorflow.python.ops import variables
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.python.training import saver
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.client import session as tf_session
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import compat
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_def_utils


SCHEMA_FILE = 'schema.json'
NUMERICAL_ANALYSIS = 'stats.json'
CATEGORICAL_ANALYSIS = 'vocab_%s.csv'


# Constants for the Prediction Graph fetch tensors.
PG_TARGET = 'target'  # from input

PG_REGRESSION_PREDICTED_TARGET = 'predicted'

PG_CLASSIFICATION_FIRST_LABEL = 'predicted'
PG_CLASSIFICATION_FIRST_SCORE = 'score'
PG_CLASSIFICATION_LABEL_TEMPLATE = 'predicted_%s'
PG_CLASSIFICATION_SCORE_TEMPLATE = 'score_%s'


class NotFittedError(ValueError):
    pass

# ==============================================================================
# Functions for saving the exported graphs.
# ==============================================================================


def _recursive_copy(src_dir, dest_dir):
  """Copy the contents of src_dir into the folder dest_dir.
  Args:
    src_dir: gsc or local path.
    dest_dir: gcs or local path.
  When called, dest_dir should exist.
  """
  src_dir = python_portable_string(src_dir)
  dest_dir = python_portable_string(dest_dir)

  file_io.recursive_create_dir(dest_dir)
  for file_name in file_io.list_directory(src_dir):
    old_path = os.path.join(src_dir, file_name)
    new_path = os.path.join(dest_dir, file_name)

    if file_io.is_directory(old_path):
      _recursive_copy(old_path, new_path)
    else:
      file_io.copy(old_path, new_path, overwrite=True)


def serving_from_csv_input(train_config, args, keep_target):
  """Read the input features from a placeholder csv string tensor."""
  examples = tf.placeholder(
      dtype=tf.string,
      shape=(None,),
      name='csv_input_string')

  features = parse_example_tensor(examples=examples,
                                  train_config=train_config,
                                  keep_target=keep_target)

  if keep_target:
    target = features.pop(train_config['target_column'])
  else:
    target = None
  features, target = preprocess_input(
      features=features,
      target=target,
      train_config=train_config,
      preprocess_output_dir=args.preprocess_output_dir,
      model_type=args.model_type)

  return input_fn_utils.InputFnOps(features,
                                   target,
                                   {'csv_line': examples}
                                   )


def make_output_tensors(train_config, args, input_ops, model_fn_ops, keep_target=True):
    target_name = train_config['target_column']
    key_name = train_config['key_column']

    outputs = {}
    outputs[key_name] = tf.squeeze(input_ops.features[key_name])

    if is_classification_model(args.model_type):

      # build maps from ints to the origional categorical strings.
      string_value = get_vocabulary(args.preprocess_output_dir, target_name)
      table = tf.contrib.lookup.index_to_string_table_from_tensor(
          mapping=string_value,
          default_value='UNKNOWN')

      # Get the label of the input target.
      if keep_target:
        input_target_label = table.lookup(input_ops.labels)
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
        outputs[PG_TARGET] = tf.squeeze(input_ops.labels)

      scores = model_fn_ops.predictions['scores']
      outputs[PG_REGRESSION_PREDICTED_TARGET] = tf.squeeze(scores)

    return outputs


def make_export_strategy(train_config, args, keep_target, assets_extra=None):
  def export_fn(estimator, export_dir_base, checkpoint_path=None, eval_result=None):
    with ops.Graph().as_default() as g:
      contrib_variables.create_global_step(g)

      input_ops = serving_from_csv_input(train_config, args, keep_target)
      model_fn_ops = estimator._call_model_fn(input_ops.features,
                                              None,
                                              model_fn_lib.ModeKeys.INFER)
      output_fetch_tensors = make_output_tensors(
          train_config=train_config,
          args=args,
          input_ops=input_ops,
          model_fn_ops=model_fn_ops,
          keep_target=keep_target)

      signature_def_map = {
        'serving_default': signature_def_utils.predict_signature_def(input_ops.default_inputs,
                                                                     output_fetch_tensors)
      }

      if not checkpoint_path:
        # Locate the latest checkpoint
        checkpoint_path = saver.latest_checkpoint(estimator._model_dir)
      if not checkpoint_path:
        raise NotFittedError("Couldn't find trained model at %s."
                             % estimator._model_dir)

      export_dir = saved_model_export_utils.get_timestamped_export_dir(
          export_dir_base)

      with tf_session.Session('') as session:
        # variables.initialize_local_variables()
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
          gfile.MakeDirs(dest_path)
          gfile.Copy(source, dest_absolute)

    # only keep the last 3 models
    saved_model_export_utils.garbage_collect_exports(
        python_portable_string(export_dir_base),
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
    _recursive_copy(export_dir, final_dir)

    return export_dir

  if keep_target:
    intermediate_dir = 'intermediate_evaluation_models'
  else:
    intermediate_dir = 'intermediate_prediction_models'

  return export_strategy.ExportStrategy(intermediate_dir, export_fn)


# ==============================================================================
# Reading the input csv files and parsing its output into tensors.
# ==============================================================================


def parse_example_tensor(examples, train_config, keep_target):
  """Read the csv files.

  Args:
    examples: string tensor
    train_config: training config
    keep_target: if true, the target column is expected to exist and it is
        returned in the features dict.

  Returns:
    Dict of feature_name to tensor. Target feature is in the dict.
  """

  csv_header = []
  if keep_target:
    csv_header = train_config['csv_header']
  else:
    csv_header = [name for name in train_config['csv_header']
                  if name != train_config['target_column']]

  # record_defaults are used by tf.decode_csv to insert defaults, and to infer
  # the datatype.
  record_defaults = [[train_config['csv_defaults'][name]]
                     for name in csv_header]
  tensors = tf.decode_csv(examples, record_defaults, name='csv_to_tensors')

  # I'm not really sure why expand_dims needs to be called. If using regression
  # models, it errors without it.
  tensors = [tf.expand_dims(x, axis=1) for x in tensors]

  tensor_dict = dict(zip(csv_header, tensors))
  return tensor_dict


def read_examples(input_files, batch_size, shuffle, num_epochs=None):
  """Creates readers and queues for reading example protos."""
  files = []
  for e in input_files:
    for path in e.split(','):
      files.extend(file_io.get_matching_files(path))
  thread_count = multiprocessing.cpu_count()

  # The minimum number of instances in a queue from which examples are drawn
  # randomly. The larger this number, the more randomness at the expense of
  # higher memory requirements.
  min_after_dequeue = 1000

  # When batching data, the queue's capacity will be larger than the batch_size
  # by some factor. The recommended formula is (num_threads + a small safety
  # margin). For now, we use a single thread for reading, so this can be small.
  queue_size_multiplier = thread_count + 3

  # Convert num_epochs == 0 -> num_epochs is None, if necessary
  num_epochs = num_epochs or None

  # Build a queue of the filenames to be read.
  filename_queue = tf.train.string_input_producer(files, num_epochs, shuffle)

  example_id, encoded_example = tf.TextLineReader().read_up_to(
      filename_queue, batch_size)

  if shuffle:
    capacity = min_after_dequeue + queue_size_multiplier * batch_size
    return tf.train.shuffle_batch(
        [example_id, encoded_example],
        batch_size,
        capacity,
        min_after_dequeue,
        enqueue_many=True,
        num_threads=thread_count)

  else:
    capacity = queue_size_multiplier * batch_size
    return tf.train.batch(
        [example_id, encoded_example],
        batch_size,
        capacity=capacity,
        enqueue_many=True,
        num_threads=thread_count)


# ==============================================================================
# Building the TF learn estimators
# ==============================================================================


def get_estimator(output_dir, train_config, args):
  """Returns a tf learn estimator.

  We only support {DNN, Linear}Regressor and {DNN, Linear}Classifier. This is
  controlled by the values of model_type in the args.

  Args:
    output_dir: Modes are saved into outputdir/train
    train_config: our training config
    args: command line parameters

  Returns:
    TF lean estimator

  Raises:
    ValueError: if config is wrong.
  """

  # Check the requested mode fits the preprocessed data.
  target_name = train_config['target_column']
  if is_classification_model(args.model_type) and target_name not in \
          train_config['categorical_columns']:
    raise ValueError('When using a classification model, the target must be a '
                     'categorical variable.')
  if is_regression_model(args.model_type) and target_name not in \
          train_config['numerical_columns']:
    raise ValueError('When using a regression model, the target must be a '
                     'numerical variable.')

  # Check layers used for dnn models.
  if is_dnn_model(args.model_type) and not args.layer_sizes:
    raise ValueError('--layer-size* must be used with DNN models')
  if is_linear_model(args.model_type) and args.layer_sizes:
    raise ValueError('--layer-size* cannot be used with linear models')

  # Build tf.learn features
  feature_columns = _tflearn_features(train_config, args)

  # Set how often to run checkpointing in terms of time.
  config = tf.contrib.learn.RunConfig(
      save_checkpoints_secs=args.save_checkpoints_secs)

  train_dir = os.path.join(output_dir, 'train')
  if args.model_type == 'dnn_regression':
    estimator = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=args.layer_sizes,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.AdamOptimizer(
            args.learning_rate, epsilon=args.epsilon))
  elif args.model_type == 'linear_regression':
    estimator = tf.contrib.learn.LinearRegressor(
        feature_columns=feature_columns,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.AdamOptimizer(
            args.learning_rate, epsilon=args.epsilon))
  elif args.model_type == 'dnn_classification':
    estimator = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=args.layer_sizes,
        n_classes=train_config['vocab_stats'][target_name]['n_classes'],
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.AdamOptimizer(
            args.learning_rate, epsilon=args.epsilon))
  elif args.model_type == 'linear_classification':
    estimator = tf.contrib.learn.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=train_config['vocab_stats'][target_name]['n_classes'],
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.AdamOptimizer(
            args.learning_rate, epsilon=args.epsilon))
  else:
    raise ValueError('bad --model-type value')

  return estimator


def preprocess_input(features, target, train_config, preprocess_output_dir,
                     model_type):
  """Perform some transformations after reading in the input tensors.

  Args:
    features: dict of feature_name to tensor
    target: tensor
    train_config: our training config object
    preprocess_output_dir: folder should contain the vocab files.
    model_type: the tf model type.

  Raises:
    ValueError: if wrong transforms are used

  Returns:
    New features dict and new target tensor.
  """

  target_name = train_config['target_column']
  key_name = train_config['key_column']

  # Do the numerical transforms.
  # Numerical transforms supported for regression/classification
  # 1) num -> do nothing (identity, default)
  # 2) num -> scale to -1, 1 (scale)
  # 3) num -> scale to -a, a (scale with value parameter)
  with tf.name_scope('numerical_feature_preprocess'):
    if train_config['numerical_columns']:
      numerical_analysis_file = os.path.join(preprocess_output_dir,
                                             NUMERICAL_ANALYSIS)
      if not file_io.file_exists(numerical_analysis_file):
        raise ValueError('File %s not found in %s' %
                         (NUMERICAL_ANALYSIS, preprocess_output_dir))

      numerical_anlysis = json.loads(
          python_portable_string(
              file_io.read_file_to_string(numerical_analysis_file)))

      for name in train_config['numerical_columns']:
        if name == target_name or name == key_name:
          continue

        transform_config = train_config['transforms'].get(name, {})
        transform_name = transform_config.get('transform', None)
        if transform_name == 'scale':
          value = float(transform_config.get('value', 1.0))
          features[name] = _scale_tensor(
              features[name],
              range_min=numerical_anlysis[name]['min'],
              range_max=numerical_anlysis[name]['max'],
              scale_min=-value,
              scale_max=value)
        elif transform_name == 'identity' or transform_name is None:
          pass
        else:
          raise ValueError(('For numerical variables, only scale '
                            'and identity are supported: '
                            'Error for %s') % name)

  # Do target transform if it exists.
  if target is not None:
    with tf.name_scope('target_feature_preprocess'):
      if target_name in train_config['categorical_columns']:
        labels = train_config['vocab_stats'][target_name]['labels']
        table = tf.contrib.lookup.string_to_index_table_from_tensor(labels)
        target = table.lookup(target)
        # target = tf.contrib.lookup.string_to_index(target, labels)

  # Do categorical transforms. Only apply vocab mapping. The real
  # transforms are done with tf learn column features.
  with tf.name_scope('categorical_feature_preprocess'):
    for name in train_config['categorical_columns']:
      if name == key_name or name == target_name:
        continue
      transform_config = train_config['transforms'].get(name, {})
      transform_name = transform_config.get('transform', None)

      if is_dnn_model(model_type):
        if transform_name == 'embedding' or transform_name == 'one_hot' or transform_name is None:
          map_vocab = True
        else:
          raise ValueError('Unknown transform %s' % transform_name)
      elif is_linear_model(model_type):
        if (transform_name == 'one_hot' or transform_name is None):
          map_vocab = True
        elif transform_name == 'embedding':
          map_vocab = False
        else:
          raise ValueError('Unknown transform %s' % transform_name)
      if map_vocab:
        labels = train_config['vocab_stats'][name]['labels']
        table = tf.contrib.lookup.string_to_index_table_from_tensor(labels)
        features[name] = table.lookup(features[name])

  return features, target


def _scale_tensor(tensor, range_min, range_max, scale_min, scale_max):
  """Scale a tensor to scale_min to scale_max.

  Args:
    tensor: input tensor. Should be a numerical tensor.
    range_min: min expected value for this feature/tensor.
    range_max: max expected Value.
    scale_min: new expected min value.
    scale_max: new expected max value.

  Returns:
    scaled tensor.
  """
  if range_min == range_max:
    return tensor

  float_tensor = tf.to_float(tensor)
  scaled_tensor = tf.divide((tf.subtract(float_tensor, range_min) *
                             tf.constant(float(scale_max - scale_min))),
                            tf.constant(float(range_max - range_min)))
  shifted_tensor = scaled_tensor + tf.constant(float(scale_min))

  return shifted_tensor


def _tflearn_features(train_config, args):
  """Builds the tf.learn feature list.

  All numerical features are just given real_valued_column because all the
  preprocessing transformations are done in preprocess_input. Categoriacl
  features are processed here depending if the vocab map (from string to int)
  was applied in preprocess_input.

  Args:
    train_config: our train config object
    args: command line args.

  Returns:
    List of TF lean feature columns.

  Raises:
    ValueError: if wrong transforms are used for the model type.
  """
  feature_columns = []
  target_name = train_config['target_column']
  key_name = train_config['key_column']

  for name in train_config['numerical_columns']:
    if name != target_name and name != key_name:
      feature_columns.append(tf.contrib.layers.real_valued_column(
          name,
          dimension=1))

      # Supported transforms:
      # for DNN
      # 1) string -> make int -> embedding (embedding)
      # 2) string -> make int -> one_hot (one_hot, default)
      # for linear
      # 1) string -> sparse_column_with_hash_bucket (embedding)
      # 2) string -> make int -> sparse_column_with_integerized_feature (one_hot, default)
      # It is unfortunate that tf.layers has different feature transforms if the
      # model is linear or DNN. This pacakge should not expose to the user that
      # we are using tf.layers. It is crazy that DNN models support more feature
      # types (like string -> hash sparse column -> embedding)
  for name in train_config['categorical_columns']:
    if name != target_name and name != key_name:
      transform_config = train_config['transforms'].get(name, {})
      transform_name = transform_config.get('transform', None)

      if is_dnn_model(args.model_type):
        if transform_name == 'embedding':
          sparse = tf.contrib.layers.sparse_column_with_integerized_feature(
              name,
              bucket_size=train_config['vocab_stats'][name]['n_classes'])
          learn_feature = tf.contrib.layers.embedding_column(
              sparse,
              dimension=transform_config['embedding_dim'])
        elif transform_name == 'one_hot' or transform_name is None:
          sparse = tf.contrib.layers.sparse_column_with_integerized_feature(
              name,
              bucket_size=train_config['vocab_stats'][name]['n_classes'])
          learn_feature = tf.contrib.layers.one_hot_column(sparse)
        else:
          raise ValueError(('Unknown transform name. Only \'embedding\' '
                            'and \'one_hot\' transforms are supported. Got %s')
                           % transform_name)
      elif is_linear_model(args.model_type):
        if transform_name == 'one_hot' or transform_name is None:
          learn_feature = tf.contrib.layers.sparse_column_with_integerized_feature(
              name,
              bucket_size=train_config['vocab_stats'][name]['n_classes'])
        elif transform_name == 'embedding':
          learn_feature = tf.contrib.layers.sparse_column_with_hash_bucket(
              name,
              hash_bucket_size=transform_config['embedding_dim'])
        else:
          raise ValueError(('Unknown transform name. Only \'embedding\' '
                            'and \'one_hot\' transforms are supported. Got %s')
                           % transform_name)

      # Save the feature
      feature_columns.append(learn_feature)
  return feature_columns


# ==============================================================================
# Functions for dealing with the parameter files.
# ==============================================================================


def get_vocabulary(preprocess_output_dir, name):
  """Loads the vocabulary file as a list of strings.

  Args:
    preprocess_output_dir: Should contain the file CATEGORICAL_ANALYSIS % name.
    name: name of the csv column.

  Returns:
    List of strings.

  Raises:
    ValueError: if file is missing.
  """
  vocab_file = os.path.join(preprocess_output_dir, CATEGORICAL_ANALYSIS % name)
  if not file_io.file_exists(vocab_file):
    raise ValueError('File %s not found in %s' %
                     (CATEGORICAL_ANALYSIS % name, preprocess_output_dir))

  labels = python_portable_string(
      file_io.read_file_to_string(vocab_file)).split('\n')
  label_values = [x for x in labels if x]  # remove empty lines

  return label_values


def merge_metadata(preprocess_output_dir, transforms_file):
  """Merge schema, analysis, and transforms files into one python object.

  Args:
    preprocess_output_dir: the output folder of preprocessing. Should contain
        the schema, and the numerical and categorical
        analysis files.
    transforms_file: the training transforms file.

  Returns:
    A dict in the form
    {
      csv_header: [name1, name2, ...],
      csv_defaults: {name1: value, name2: value},
      key_column: name,
      target_column: name,
      categorical_columns: []
      numerical_columns: []
      transforms: { name1: {transform: scale, value: 2},
                    name2: {transform: embedding, dim: 50}, ...
                  }
      vocab_stats: { name3: {n_classes: 23, labels: ['1', '2', ..., '23']},
                     name4: {n_classes: 102, labels: ['red', 'blue', ...]}}
    }

  Raises:
    ValueError: if one of the input metadata files is wrong.
  """
  numerical_anlysis_file = os.path.join(preprocess_output_dir,
                                        NUMERICAL_ANALYSIS)
  schema_file = os.path.join(preprocess_output_dir, SCHEMA_FILE)

  numerical_anlysis = json.loads(
      python_portable_string(
          file_io.read_file_to_string(numerical_anlysis_file)))
  schema = json.loads(
      python_portable_string(file_io.read_file_to_string(schema_file)))
  transforms = json.loads(
      python_portable_string(file_io.read_file_to_string(transforms_file)))

  result_dict = {}
  result_dict['csv_header'] = [col_schema['name'] for col_schema in schema]
  result_dict['key_column'] = None
  result_dict['target_column'] = None
  result_dict['categorical_columns'] = []
  result_dict['numerical_columns'] = []
  result_dict['transforms'] = {}
  result_dict['csv_defaults'] = {}
  result_dict['vocab_stats'] = {}

  # get key column.
  for name, trans_config in six.iteritems(transforms):
    if trans_config.get('transform', None) == 'key':
      result_dict['key_column'] = name
      break
  if result_dict['key_column'] is None:
    raise ValueError('Key transform missing form transfroms file.')

  # get target column.
  result_dict['target_column'] = schema[0]['name']
  for name, trans_config in six.iteritems(transforms):
    if trans_config.get('transform', None) == 'target':
      result_dict['target_column'] = name
      break
  if result_dict['target_column'] is None:
    raise ValueError('Target transform missing from transforms file.')

  # Get the numerical/categorical columns.
  for col_schema in schema:
    col_name = col_schema['name']
    col_type = col_schema['type'].lower()
    if col_name == result_dict['key_column']:
      continue

    if col_type == 'string':
      result_dict['categorical_columns'].append(col_name)
    elif col_type == 'integer' or col_type == 'float':
      result_dict['numerical_columns'].append(col_name)
    else:
      raise ValueError('Unsupported schema type %s' % col_type)

  # Get the transforms.
  for name, trans_config in six.iteritems(transforms):
    if name != result_dict['target_column'] and name != result_dict['key_column']:
      result_dict['transforms'][name] = trans_config

  # Get the vocab_stats
  for name in result_dict['categorical_columns']:
    if name == result_dict['key_column']:
      continue

    label_values = get_vocabulary(preprocess_output_dir, name)
    if name != result_dict['target_column'] and '' not in label_values:
      label_values.append('')  # append a 'missing' label.
    n_classes = len(label_values)
    result_dict['vocab_stats'][name] = {'n_classes': n_classes,
                                        'labels': label_values}

  # Get the csv_defaults
  for col_schema in schema:
    name = col_schema['name']
    col_type = col_schema['type'].lower()
    default = transforms.get(name, {}).get('default', None)

    if name == result_dict['target_column']:
      if name in result_dict['numerical_columns']:
        default = float(default or 0.0)
      else:
        default = default or ''
    elif name == result_dict['key_column']:
      if col_type == 'string':
        default = str(default or '')
      elif col_type == 'float':
        default = float(default or 0.0)
      else:
        default = int(default or 0)
    else:
      if col_type == 'string':
        default = str(default or '')
        if default not in result_dict['vocab_stats'][name]['labels']:
          raise ValueError('Default %s is not in the vocab for %s' %
                           (default, name))
      else:
        default = float(default or numerical_anlysis[name]['mean'])

    result_dict['csv_defaults'][name] = default

  validate_metadata(result_dict)
  return result_dict


def validate_metadata(train_config):
  """Perform some checks that the trainig config is correct.

  Args:
    train_config: train config as produced by merge_metadata()

  Raises:
    ValueError: if columns look wrong.
  """

  # Make sure we have a default for every column
  if len(train_config['csv_header']) != len(train_config['csv_defaults']):
    raise ValueError('Unequal number of columns in input features file and '
                     'schema file.')

  # Check there are no missing columns. sorted_colums has two copies of the
  # target column because the target column is also listed in
  # categorical_columns or numerical_columns.
  sorted_columns = sorted(train_config['csv_header'] +
                          [train_config['target_column']])

  sorted_columns2 = sorted(train_config['categorical_columns'] +
                           train_config['numerical_columns'] +
                           [train_config['key_column']] +
                           [train_config['target_column']])
  if sorted_columns2 != sorted_columns:
    raise ValueError('Each csv header must be a numerical/categorical type, a '
                     ' key, or a target.')


def is_linear_model(model_type):
  return model_type.startswith('linear_')


def is_dnn_model(model_type):
  return model_type.startswith('dnn_')


def is_regression_model(model_type):
  return model_type.endswith('_regression')


def is_classification_model(model_type):
  return model_type.endswith('_classification')


# Note that this function exists in google.datalab.utils, but that is not
# installed on the training workers.
def python_portable_string(string, encoding='utf-8'):
  """Converts bytes into a string type.

  Valid string types are retuned without modification. So in Python 2, type str
  and unicode are not converted.

  In Python 3, type bytes is converted to type str (unicode)
  """
  if isinstance(string, six.string_types):
    return string

  if six.PY3:
    return string.decode(encoding)

  raise ValueError('Unsupported type %s' % str(type(string)))
