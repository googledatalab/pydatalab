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


"""Reusable utility functions.
"""

import collections
import multiprocessing
import os

import tensorflow as tf
from tensorflow.python.lib.io import file_io


_DEFAULT_CHECKPOINT_GSURL = 'gs://cloud-ml-data/img/flower_photos/inception_v3_2016_08_28.ckpt'


def is_in_IPython():
  try:
    import IPython # noqa
    return True
  except ImportError:
    return False


def default_project():
  from google.datalab import Context
  return Context.default().project_id


def _get_latest_data_dir(input_dir):
  latest_file = os.path.join(input_dir, 'latest')
  if not file_io.file_exists(latest_file):
    raise Exception(('Cannot find "latest" file in "%s". ' +
                    'Please use a preprocessing output dir.') % input_dir)
  with file_io.FileIO(latest_file, 'r') as f:
    dir_name = f.read().rstrip()
  return os.path.join(input_dir, dir_name)


def get_train_eval_files(input_dir):
  """Get preprocessed training and eval files."""
  data_dir = _get_latest_data_dir(input_dir)
  train_pattern = os.path.join(data_dir, 'train*.tfrecord.gz')
  eval_pattern = os.path.join(data_dir, 'eval*.tfrecord.gz')
  train_files = file_io.get_matching_files(train_pattern)
  eval_files = file_io.get_matching_files(eval_pattern)
  return train_files, eval_files


def get_labels(input_dir):
  """Get a list of labels from preprocessed output dir."""
  data_dir = _get_latest_data_dir(input_dir)
  labels_file = os.path.join(data_dir, 'labels')
  with file_io.FileIO(labels_file, 'r') as f:
    labels = f.read().rstrip().split('\n')
  return labels


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


def override_if_not_in_args(flag, argument, args):
  """Checks if flags is in args, and if not it adds the flag to args."""
  if flag not in args:
    args.extend([flag, argument])


def loss(loss_value):
  """Calculates aggregated mean loss."""
  total_loss = tf.Variable(0.0, False)
  loss_count = tf.Variable(0, False)
  total_loss_update = tf.assign_add(total_loss, loss_value)
  loss_count_update = tf.assign_add(loss_count, 1)
  loss_op = total_loss / tf.cast(loss_count, tf.float32)
  return [total_loss_update, loss_count_update], loss_op


def accuracy(logits, labels):
  """Calculates aggregated accuracy."""
  is_correct = tf.nn.in_top_k(logits, labels, 1)
  correct = tf.reduce_sum(tf.cast(is_correct, tf.int32))
  incorrect = tf.reduce_sum(tf.cast(tf.logical_not(is_correct), tf.int32))
  correct_count = tf.Variable(0, False)
  incorrect_count = tf.Variable(0, False)
  correct_count_update = tf.assign_add(correct_count, correct)
  incorrect_count_update = tf.assign_add(incorrect_count, incorrect)
  accuracy_op = tf.cast(correct_count, tf.float32) / tf.cast(
      correct_count + incorrect_count, tf.float32)
  return [correct_count_update, incorrect_count_update], accuracy_op


def check_dataset(dataset, mode):
  """Validate we have a good dataset."""

  names = [x['name'] for x in dataset.schema]
  types = [x['type'] for x in dataset.schema]
  if mode == 'train':
    if (set(['image_url', 'label']) != set(names) or any(t != 'STRING' for t in types)):
      raise ValueError('Invalid dataset. Expect only "image_url,label" STRING columns.')
  else:
    if (set(['image_url']) != set(names) and set(['image_url', 'label']) != set(names)) or \
            any(t != 'STRING' for t in types):
      raise ValueError('Invalid dataset. Expect only "image_url" or "image_url,label" ' +
                       'STRING columns.')


def get_sources_from_dataset(p, dataset, mode):
  """get pcollection from dataset."""

  import apache_beam as beam
  import csv
  from google.datalab.ml import CsvDataSet, BigQueryDataSet

  check_dataset(dataset, mode)
  if type(dataset) is CsvDataSet:
    source_list = []
    for ii, input_path in enumerate(dataset.files):
      source_list.append(p | 'Read from Csv %d (%s)' % (ii, mode) >>
                         beam.io.ReadFromText(input_path, strip_trailing_newlines=True))
    return (source_list |
            'Flatten Sources (%s)' % mode >>
            beam.Flatten() |
            'Create Dict from Csv (%s)' % mode >>
            beam.Map(lambda line: csv.DictReader([line], fieldnames=['image_url',
                                                                     'label']).next()))
  elif type(dataset) is BigQueryDataSet:
    bq_source = (beam.io.BigQuerySource(table=dataset.table) if dataset.table is not None else
                 beam.io.BigQuerySource(query=dataset.query))
    return p | 'Read source from BigQuery (%s)' % mode >> beam.io.Read(bq_source)
  else:
    raise ValueError('Invalid DataSet. Expect CsvDataSet or BigQueryDataSet')


def decode_and_resize(image_str_tensor):
  """Decodes jpeg string, resizes it and returns a uint8 tensor."""

  # These constants are set by Inception v3's expectations.
  height = 299
  width = 299
  channels = 3

  image = tf.image.decode_jpeg(image_str_tensor, channels=channels)
  # Note resize expects a batch_size, but tf_map supresses that index,
  # thus we have to expand then squeeze.  Resize returns float32 in the
  # range [0, uint8_max]
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
  image = tf.squeeze(image, squeeze_dims=[0])
  image = tf.cast(image, dtype=tf.uint8)
  return image


def resize_image(image_str_tensor):
  """Decodes jpeg string, resizes it and re-encode it to jpeg."""

  image = decode_and_resize(image_str_tensor)
  image = tf.image.encode_jpeg(image, quality=100)
  return image


def load_images(image_files, resize=True):
  """Load images from files and optionally resize it."""

  images = []
  for image_file in image_files:
    with file_io.FileIO(image_file, 'r') as ff:
      images.append(ff.read())
  if resize is False:
    return images

  # To resize, run a tf session so we can reuse 'decode_and_resize()'
  # which is used in prediction graph. This makes sure we don't lose
  # any quality in prediction, while decreasing the size of the images
  # submitted to the model over network.
  image_str_tensor = tf.placeholder(tf.string, shape=[None])
  image = tf.map_fn(resize_image, image_str_tensor, back_prop=False)
  feed_dict = collections.defaultdict(list)
  feed_dict[image_str_tensor.name] = images
  with tf.Session() as sess:
    images_resized = sess.run(image, feed_dict=feed_dict)
  return images_resized


def process_prediction_results(results, show_image):
  """Create DataFrames out of prediction results, and display images in IPython if requested."""

  import pandas as pd

  if (is_in_IPython() and show_image is True):
    import IPython
    for image_url, image, label_and_score in results:
      IPython.display.display_html('<p style="font-size:28px">%s(%.5f)</p>' % label_and_score,
                                   raw=True)
      IPython.display.display(IPython.display.Image(data=image))
  result_dict = [{'image_url': url, 'label': r[0], 'score': r[1]} for url, _, r in results]
  return pd.DataFrame(result_dict)


def repackage_to_staging(output_path):
  """Repackage it from local installed location and copy it to GCS."""

  import google.datalab.ml as ml

  # Find the package root. __file__ is under [package_root]/mltoolbox/image/classification.
  package_root = os.path.join(os.path.dirname(__file__), '../../../')
  # We deploy setup.py in the same dir for repackaging purpose.
  setup_py = os.path.join(os.path.dirname(__file__), 'setup.py')
  staging_package_url = os.path.join(output_path, 'staging', 'image_classification.tar.gz')
  ml.package_and_copy(package_root, setup_py, staging_package_url)
  return staging_package_url
