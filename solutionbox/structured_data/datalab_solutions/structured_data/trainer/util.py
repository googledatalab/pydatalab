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




def produce_feature_columns(metadata, config):
  """Produces a list of Tensorflow columns.

  Args:
    metadata: dict from the preprocessing metadata.
    config: dict from the transforms file.

  Returns:
    List of Tensorflow feature columns.
  """
  
  feature_columns = []
  # Extract the numerical features. 
  if 'numerical' in config:
    for name, transform_config in config['numerical'].iteritems():
      # There is no other TF transfroms for numerical columns.
      feature_columns.append(
          tf.contrib.layers.real_valued_column(
              name,
              dimension=metadata.features[name]['size']))
      # TODO(brandondutra) allow real_valued vectors? For now force only scalars.
      assert 1 == metadata.features[name]['size']

  # Extrace the categorical features
  if 'categorical' in config:
    for name, transform_config in config['categorical'].iteritems():
      transform = transform_config['transform']
      if config['model_type'] == 'linear' and transform != 'one_hot':
        print('ERROR: only one_hot transfroms are supported in linear models')
        sys.exit(1)

      if transform == 'one_hot':
        # Preprocessing built a vocab using 0, 1, ..., N as the new classes.
        N = metadata.features[name]['size']
        sparse_column = tf.contrib.layers.sparse_column_with_integerized_feature(
            column_name=name,
            bucket_size=N+1)
        if config['model_type'] == 'linear':
          feature_columns.append(sparse_column)
        else:
          feature_columns.append(
              tf.contrib.layers.one_hot_column(sparse_column))
      elif transform == 'embedding':
        # Preprocessing built a vocab using 0, 1, ..., N as the new classes.
        N = metadata.features[name]['size']
        dim = transform_config['dimension']
        sparse_column = tf.contrib.layers.sparse_column_with_integerized_feature(
            column_name=name,
            bucket_size=N+1)
        feature_columns.append(
            tf.contrib.layers.embedding_column(sparse_column, dim))
      else:
        print('ERROR: unkown categorical transform name %s in %s' % (transform, str(transform_config)))
        sys.exit(1)

  return feature_columns


def parse_example_tensor(examples, mode, metadata, transform_config):
  if mode == 'training':
    raw_tensor = ml.features.FeatureMetadata.parse_features(metadata, examples,
                                                            keep_target=True)
  elif mode == 'prediction':
    raw_tensor = ml.features.FeatureMetadata.parse_features(metadata, examples,
                                                            keep_target=False)
  return raw_tensor
  # dtype_mapping = {
  #     'bytes': tf.string,
  #     'float': tf.float32,
  #     'int64': tf.int64
  # }

  # example_schema = {}
  # if 'numerical' in transform_config:
  #   for name, _ in transform_config['numerical'].iteritems():
  #     size = 1 #metadata.features[name]['size']
  #     dtype = dtype_mapping[metadata.features[name]['dtype']]
  #     example_schema[name] = tf.FixedLenFeature(shape=[size], dtype=dtype)

  # if 'categorical' in transform_config:
  #   for name, _ in transform_config['categorical'].iteritems():
  #     size = 1 #metadata.features[name]['size']
  #     dtype = dtype_mapping[metadata.features[name]['dtype']]
  #     example_schema[name] = tf.FixedLenFeature(shape=[size], dtype=dtype)


  # if mode == 'training':
  #   target_name = transform_config['target_column']
  #   size = 1 #metadata.features[target_name]['size']
  #   dtype = dtype_mapping[metadata.features[target_name]['dtype']]
  #   example_schema[target_name] = tf.FixedLenFeature(shape=[size], dtype=dtype)
  # elif mode == 'prediction':
  #   key_name = transform_config['key_column']
  #   size = 1 #metadata.features[key_name]['size']
  #   dtype = dtype_mapping[metadata.features[key_name]['dtype']]
  #   example_schema[key_name] = tf.FixedLenFeature(shape=[size], dtype=dtype)
  # else:
  #   print('ERROR: unknown mode type')
  #   sys.exit(1)

  # return tf.parse_example(examples, example_schema)
