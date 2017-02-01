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

import multiprocessing
import os
import sys
import json
import google.cloud.ml as ml
import pandas as pd
from StringIO import StringIO

import tensorflow as tf
from tensorflow.python.lib.io import file_io
import google.cloud.ml as ml

INPUT_FEATURES_FILE = 'input_features.json'
SCHEMA_FILE = 'schema.json'
NUMERICAL_ANALYSIS = 'numerical_analysis.json'
CATEGORICAL_ANALYSIS = 'vocab_%s.csv'


# ==============================================================================
# Exporting the last trained model to a final location
# ==============================================================================

def _copy_all(src_files, dest_dir):
  # file_io.copy does not copy files into folders directly.
  for src_file in src_files:
    file_name = os.path.basename(src_file)
    new_file_location = os.path.join(dest_dir, file_name)
    file_io.copy(src_file, new_file_location)


def _recursive_copy(src_dir, dest_dir):
  """Copy the contents of src_dir into the folder dest_dir.

  When called, dest_dir should exist.
  """
  for dir_name, sub_dirs, leaf_files in file_io.walk(src_dir):
    # copy all the files over
    for leaf_file in leaf_files:
      leaf_file_path = os.path.join(dir_name, leaf_file)
      _copy_all([leaf_file_path], dest_dir)

    # Now make all the folders.
    for sub_dir in sub_dirs:
      file_io.create_dir(os.path.join(dest_dir, sub_dir))


class ExportLastModelMonitor(tf.contrib.learn.monitors.ExportMonitor):
  """Export the last model to its final location on GCS.

  The tf.learn ExportMonitor exports the models to a location based on the last
  n steps move the exported model to a fixed location.
  """

  def __init__(self,
               output_dir,
               final_model_location,
               every_n_steps=5000,
               additional_assets=None,
               input_fn=None,
               input_feature_key=None,
               exports_to_keep=5,
               signature_fn=None,
               default_batch_size=None):
    # Export the model to a temporary location then upload to its final
    # GCS destination.
    export_dir = os.path.join(output_dir, 'intermediate_models')
    super(ExportLastModelMonitor, self).__init__(
        every_n_steps=every_n_steps,
        export_dir=export_dir,
        input_fn=input_fn,
        input_feature_key=input_feature_key,
        exports_to_keep=exports_to_keep,
        signature_fn=signature_fn,
        default_batch_size=default_batch_size)
    self._final_model_location = os.path.join(output_dir, final_model_location)
    self._additional_assets = additional_assets or []

  def end(self, session=None):
    super(ExportLastModelMonitor, self).end(session)
    # Recursively copy the last location export dir from the exporter into the
    # main export location.
    file_io.recursive_create_dir(self._final_model_location)
    _recursive_copy(self.last_export_dir, self._final_model_location)

    if self._additional_assets:
      # TODO(rhaertel): use the actual assets directory. For now, metadata.json
      # must be a sibling of the export.meta file.
      assets_dir = self._final_model_location
      file_io.create_dir(assets_dir)
      _copy_all(self._additional_assets, assets_dir)


# ==============================================================================
# Reading the input csv files and parsing its output into tensors.
# ==============================================================================

def parse_example_tensor(examples, train_config):

  record_defaults = [[train_config['csv_defaults'][name]] 
                     for name in train_config['csv_header']]
  print('record_defaults', record_defaults)
  print('*'*10)
  sys.stdout.flush()
  tensors = tf.decode_csv(examples, record_defaults, name='csv_to_tensors')

  tensor_dict = dict(zip(train_config['csv_header'], tensors))

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

  #options = tf.python_io.TFRecordOptions(
  #    compression_type=tf.python_io.TFRecordCompressionType.GZIP)
  #example_id, encoded_example = tf.TFRecordReader(options=options).read_up_to(
  #    filename_queue, batch_size)

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
    args: parseargs object
  """

  # Check the requested mode fits the preprocessed data.
  target_name = train_config['target_column']
  if (is_classification_model(args.model_type) and
     target_name not in train_config['categorical_columns']):
    raise ValueError('When using a classification model, the target must be a '
                     'categorical variable.')
  if (is_regression_model(args.model_type) and
      target_name not in train_config['numerical_columns']):
    raise ValueError('When using a regression model, the target must be a '
                     'numerical variable.')

  # Check layers used for dnn models.
  if is_dnn_model(args.model_type)  and not args.layer_sizes:
    raise ValueError('--layer_sizes must be used with DNN models')
  if is_linear_model(args.model_type) and args.layer_sizes:
    raise ValueError('--layer_sizes cannot be used with linear models')

  # Build tf.learn features
  #feature_columns = util.produce_feature_columns(
  #    schema_plus, args, is_linear_model(args.model_type))
  #feature_engineering_fn = util.produce_feature_engineering_fn(schema_plus, args)
  estimator_parameters = get_estimator_parameters(train_config, args)

  # Set how often to run checkpointing in terms of time.
  config = tf.contrib.learn.RunConfig(
      save_checkpoints_secs=args.save_checkpoints_secs)

  train_dir = os.path.join(output_dir, 'train')
  if args.model_type == 'dnn_regression':
    estimator = tf.contrib.learn.DNNRegressor(
        feature_columns=estimator_parameters['features'],
        hidden_units=args.layer_sizes,
        config=config,
        model_dir=train_dir,
        feature_engineering_fn=estimator_parameters['feature_engineering_fn'],
        optimizer=tf.train.AdamOptimizer(
          args.learning_rate, epsilon=args.epsilon))
  elif args.model_type == 'linear_regression':
    estimator = tf.contrib.learn.LinearRegressor(
        feature_columns=estimator_parameters['features'],
        config=config,
        model_dir=train_dir,
        feature_engineering_fn=estimator_parameters['feature_engineering_fn'],
        optimizer=tf.train.AdamOptimizer(
          args.learning_rate, epsilon=args.epsilon))
  elif args.model_type == 'dnn_classification':
    estimator = tf.contrib.learn.DNNClassifier(
        feature_columns=estimator_parameters['features'],
        hidden_units=args.layer_sizes,
        n_classes=estimator_parameters['n_classes'],
        config=config,
        model_dir=train_dir,
        feature_engineering_fn=estimator_parameters['feature_engineering_fn'],
        optimizer=tf.train.AdamOptimizer(
          args.learning_rate, epsilon=args.epsilon))
  elif args.model_type == 'linear_classification':
    estimator = tf.contrib.learn.LinearClassifier(
        feature_columns=estimator_parameters['features'],
        n_classes=estimator_parameters['n_classes'],
        config=config,
        model_dir=train_dir,
        feature_engineering_fn=estimator_parameters['feature_engineering_fn'],
        optimizer=tf.train.AdamOptimizer(
          args.learning_rate, epsilon=args.epsilon))
  else:
    print('ERROR: bad --model_type value')
    sys.exit(1)

  return estimator


def get_estimator_parameters(train_config, args):
  feature_columns = []
  target_name = train_config['target_column']
  key_name = train_config['key_column']

  with tf.name_scope('tflearn_features') as scope:
    for name in train_config['numerical_columns']:
      if name != target_name and name != key_name:
        feature_columns.append(tf.contrib.layers.real_valued_column(
            name,
            dimension=1))

    for name in train_config['categorical_columns']:
      if name != target_name and name != key_name:
        sparse = tf.contrib.layers.sparse_column_with_hash_bucket(
            name,
            hash_bucket_size=3)
        feature_columns.append(tf.contrib.layers.one_hot_column(sparse))


  if target_name in train_config['categorical_columns']:
    target_index_values, target_label_values = get_vocabulary(args.preprocess_output_dir, target_name)
    n_classes = len(target_label_values)

    print('target_index_values', target_index_values)
    print('target_label_values', target_label_values)

  def _feature_engineering_fn(features, target):
    with tf.name_scope('categorical_feature_engineering') as scope:
      if target_name in train_config['categorical_columns']:
        #target = tf.Print(target, [target])
        new_target = tf.contrib.lookup.string_to_index(target, target_label_values)
        #new_target = tf.Print(new_target, [new_target])
      else:
        new_target = target
    return features, new_target

  estimator_parameters = {'features': feature_columns,
          'feature_engineering_fn': _feature_engineering_fn}
  if target_name in train_config['categorical_columns']:
    estimator_parameters.update({'n_classes': n_classes})

  return estimator_parameters

# ==============================================================================
# Building the TF learn estimators
# ==============================================================================


def get_vocabulary(preprocess_output_dir, name):
  vocab_file = os.path.join(preprocess_output_dir, CATEGORICAL_ANALYSIS % name)
  if not ml.util._file.file_exists(vocab_file):
    raise ValueError('File %s not found in %s' % (CATEGORICAL_ANALYSIS % name, preprocess_output_dir))

  df = pd.read_csv(StringIO(ml.util._file.load_file(vocab_file)),
                   header=0)
  index_values = df['index'].values.tolist()
  label_values = df[name].values.tolist()

  if index_values != range(len(index_values)):
    raise ValueError('Labels should be sorted by index in file %s' % vocab_file)
  return index_values, [str(value) for value in label_values]



def merge_metadata(preprocess_output_dir, transforms_file):
  """Merge schema, input features, and transforms file into one python object.


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

    }
  """

  schema_file = os.path.join(preprocess_output_dir, SCHEMA_FILE)
  input_features_file = os.path.join(preprocess_output_dir, INPUT_FEATURES_FILE)


  schema = json.loads(ml.util._file.load_file(schema_file))
  input_features = json.loads(ml.util._file.load_file(input_features_file))
  transforms = json.loads(ml.util._file.load_file(transforms_file))

  result_dict = {}
  result_dict['csv_header'] = [schema_dict['name'] for schema_dict in schema]
  result_dict['csv_defaults'] = {}
  result_dict['key_column'] =  None
  result_dict['target_column'] =  None
  result_dict['categorical_columns'] = []
  result_dict['numerical_columns'] = []
  result_dict['transforms'] = {}

  # get key column
  for name, input_type in input_features.iteritems():
    if input_type['type'] == 'key':
      result_dict['key_column'] = name
      break
  if result_dict['key_column'] is None:
    raise ValueError('Key column missing from input features file.')

  # get target column
  for name, transform in transforms.iteritems():
    if transform.get('transform', None) == 'target':
      result_dict['target_column'] = name
      break
  if result_dict['target_column'] is None:
    raise ValueError('Target transform missing form transfroms file.')

  # Get the numerical/categorical columns.
  for schema_dict in schema:
    name = schema_dict['name']
    col_type = input_features.get(name, {}).get('type', None)

    if col_type is None:
        raise ValueError('Missing type from %s in file %s' % (
            name, input_features_file))

    if col_type == 'numerical':
      result_dict['numerical_columns'].append(name)
    elif col_type == 'categorical':
      result_dict['categorical_columns'].append(name)
    elif col_type == 'key':
      pass
    else:
      raise ValueError('unknown type %s in input featrues file.' % col_type)
      result_dict['key_column'] = name

  # Get the defaults
  for schema_dict in schema:
    name = schema_dict['name']
    default = input_features.get(name, {}).get('default', None)

    if default is None:
        raise ValueError('Missing default from %s in file %s' % (
            name, input_features_file))

    # make all numerical types floats. This means when tf.decode_csv is called,
    # float tensors and string tensors will be made.
    if name in result_dict['categorical_columns']:
      default = str(default)
    elif name in result_dict['numerical_columns']:
      default = float(default)
    else:
      default = str(default)  # key column

    result_dict['csv_defaults'].update({name: default})

  # Get the transforms
  for name, transform in transforms.iteritems():
    if transform['transform'] != 'target':
      result_dict['transforms'].update({name: transform})

  validate_metadata(result_dict)
  return result_dict

def validate_metadata(train_config):
  # Check there are no missing columns
  sorted_columns = sorted(train_config['csv_header']
                          + [train_config['target_column']])
  sorted_columns2 = sorted(train_config['categorical_columns']
                           + train_config['numerical_columns']
                           + [train_config['key_column']]
                           + [train_config['target_column']])
  if sorted_columns2 != sorted_columns:
    raise ValueError('Each csv header must be a numerical/categorical type, a '
                     ' key, or a target.')

  # Make sure we have a default for every column
  if len(train_config['csv_header']) != len(train_config['csv_defaults']):
    raise ValueError('Unequal number of columns in input features file and '
                     'schema file.')


def is_linear_model(model_type):
  return model_type.startswith('linear_')

def is_dnn_model(model_type):
  return model_type.startswith('dnn_')

def is_regression_model(model_type):
  return model_type.endswith('_regression')

def is_classification_model(model_type):
  return model_type.endswith('_classification')