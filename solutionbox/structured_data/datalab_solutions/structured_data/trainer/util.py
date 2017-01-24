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

import tensorflow as tf
from tensorflow.python.lib.io import file_io
import google.cloud.ml as ml

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

  options = tf.python_io.TFRecordOptions(
      compression_type=tf.python_io.TFRecordCompressionType.GZIP)
  example_id, encoded_example = tf.TFRecordReader(options=options).read_up_to(
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




def produce_feature_columns(metadata, transform_config, schema_config, model_type):
  """Produces a list of Tensorflow columns.

  Args:
    metadata: dict from the preprocessing metadata.
    config: dict from the transforms file.

  Returns:
    List of Tensorflow feature columns.
  """
  
  feature_columns = []
  target_column = schema_config['target_column']
  key_column = schema_config['key_column']
  # Extract the numerical features. 
  for name in schema_config.get('numerical_columns', []):
    if name == key_column or name == target_column:
      continue 
    # Numerical transforms happen in produce_feature_engineering_fn
    feature_columns.append(
        tf.contrib.layers.real_valued_column(
            name,
            dimension=metadata.features[name]['size']))
    # TODO(brandondutra) allow real_valued vectors? For now force only scalars.
    assert 1 == metadata.features[name]['size']

  # Extract the categorical features
  for name in schema_config.get('categorical_columns', []):
    if name == key_column or name == target_column:
      continue 
    transform_dict = transform_config.get(name, {})
    transform_type = transform_dict.get('transform', 'one_hot')
    
    #if config['model_type'] == 'linear' and transform != 'one_hot':
    #    print('ERROR: only one_hot transfroms are supported in linear models')
    #    sys.exit(1)

    if transform_type == 'one_hot':
      # Preprocessing built a vocab using 0, 1, ..., N as the new classes. N 
      # means 'unknown/missing'.
      N = metadata.features[name]['size']
      sparse_column = tf.contrib.layers.sparse_column_with_integerized_feature(
          column_name=name,
          bucket_size=N+1)
      if model_type == 'linear':
        feature_columns.append(sparse_column)
      else:
        feature_columns.append(tf.contrib.layers.one_hot_column(sparse_column))

    elif transform_type == 'embedding':
      # Preprocessing built a vocab using 0, 1, ..., N as the new classes.
      N = metadata.features[name]['size']
      dim = transform_dict['dimension']
      sparse_column = tf.contrib.layers.sparse_column_with_integerized_feature(
            column_name=name,
            bucket_size=N+1)
      feature_columns.append(tf.contrib.layers.embedding_column(sparse_column, 
                                                                dim))
      # TODO(brandon): check model type is dnn.
    else:
      print('ERROR: unkown categorical transform name %s in %s' % 
          (name, str(transform_type)))
      sys.exit(1)

  return feature_columns

def _scale_tensor(tensor, range_min, range_max, scale_min, scale_max):
  if range_min == range_max:
    return tensor

  float_tensor = tf.to_float(tensor)
  scaled_tensor = tf.div(
    tf.sub(float_tensor, range_min) * tf.constant(float(scale_max - scale_min)),
    tf.constant(float(range_max - range_min)))
  shifted_tensor = scaled_tensor + tf.constant(float(scale_min))

  return shifted_tensor


def produce_feature_engineering_fn(metadata, transform_config, schema_config):
  """Makes a feature_engineering_fn for transforming the numerical types. 

  This is called with the output of the 'input_fn' function, and the output of
  this function is given to tf.learn to further process. This function extracts
  the ids tensors from ml.features.FeatureMetadata.parse_features and throws
  away the values tensor.
  """

  def _feature_engineering_fn(features, target):
    target_column = schema_config['target_column']
    key_column = schema_config['key_column']

    with tf.name_scope('numerical_feature_engineering') as scope:
      new_features = {}
      for name in schema_config.get('numerical_columns', []):
        if name == key_column or name == target_column:
          continue 
        transform_dict = transform_config.get(name, {})
        transform_type = transform_dict.get('transform', 'identity')
        if transform_type == 'scale':
          range_min = metadata.columns[name]['min']
          range_max = metadata.columns[name]['max']
          new_features[name] = _scale_tensor(features[name], 
                                            range_min=range_min, 
                                            range_max=range_max, 
                                            scale_min=-1, 
                                            scale_max=1)
        elif transform_type == 'max_abs_scale':
          value = transform_dict['value']
          range_min = metadata.columns[name]['min']
          range_max = metadata.columns[name]['max']
          new_features[name] = _scale_tensor(features[name], 
                                            range_min=range_min, 
                                            range_max=range_max, 
                                            scale_min=-value, 
                                            scale_max=value)

        elif transform_type == 'identity':
          # Don't need to do anything
          pass
        else:
          print('ERROR: Unknown numerical transform %s for feature %s' % 
              (transform_type, name))
          sys.exit(1)
      features.update(new_features)
    return features, target

  return _feature_engineering_fn

def produce_feature_engineering_fnXXXXX(metadata, config):
  """Makes a feature_engineering_fn for transforming the numerical types. 

  This is called with the output of the 'input_fn' function, and the output of
  this function is given to tf.learn to further process. This function extracts
  the ids tensors from ml.features.FeatureMetadata.parse_features and throws
  away the values tensor.
  """

  def _feature_engineering_fn(features, target):
    with tf.name_scope('numerical_feature_engineering') as scope:
      new_features = {}
      if 'numerical' in config:
        for name, transform_dict in config['numerical'].iteritems():
          trans_name = transform_dict['transform']
          if trans_name == 'scale':
            range_min = metadata.columns[name]['min']
            range_max = metadata.columns[name]['max']
            new_features[name] = _scale_tensor(features[name], 
                                              range_min=range_min, 
                                              range_max=range_max, 
                                              scale_min=-1, 
                                              scale_max=1)
          elif trans_name == 'max_abs_scale':
            value = transform_dict['value']
            range_min = metadata.columns[name]['min']
            range_max = metadata.columns[name]['max']
            new_features[name] = _scale_tensor(features[name], 
                                              range_min=range_min, 
                                              range_max=range_max, 
                                              scale_min=-value, 
                                              scale_max=value)

          elif trans_name == 'identity':
            # Don't need to do anything
            pass
          else:
            print('ERROR: Unknown numerical transform %s for feature %s' % 
                (trans_name, name))
            sys.exit(1)
      features.update(new_features)
    return features, target

  return _feature_engineering_fn


def parse_example_tensor(examples, mode, metadata, schema_config):
  if mode == 'training':
    features = ml.features.FeatureMetadata.parse_features(metadata, examples,
                                                          keep_target=True) 
  elif mode == 'prediction':
    features = ml.features.FeatureMetadata.parse_features(metadata, examples,
                                                          keep_target=False) 
  else:
    print('ERROR: unknown mode')
    sys.exit(1)
  
  new_features = {}
  for name in schema_config.get('categorical_columns', []):
    if name == schema_config['key_column'] or name == schema_config['target_column']:
      continue
    new_features[name] = features[name]['ids']

  features.update(new_features)

  return features

