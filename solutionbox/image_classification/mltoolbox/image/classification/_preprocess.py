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


"""Preprocess pipeline implementation with Cloud DataFlow.
"""


import apache_beam as beam
from apache_beam.io import fileio
from apache_beam.io import tfrecordio
from apache_beam.metrics import Metrics
import cStringIO
import logging
import os
from PIL import Image
import tensorflow as tf

from . import _inceptionlib
from . import _util


slim = tf.contrib.slim

error_count = Metrics.counter('main', 'errorCount')
rows_count = Metrics.counter('main', 'rowsCount')
skipped_empty_line = Metrics.counter('main', 'skippedEmptyLine')
embedding_good = Metrics.counter('main', 'embedding_good')
embedding_bad = Metrics.counter('main', 'embedding_bad')
incompatible_image = Metrics.counter('main', 'incompatible_image')
invalid_uri = Metrics.counter('main', 'invalid_file_name')
unlabeled_image = Metrics.counter('main', 'unlabeled_image')


class ExtractLabelIdsDoFn(beam.DoFn):
  """Extracts (uri, label_ids) tuples from CSV rows.
  """

  def start_bundle(self, context=None):
    self.label_to_id_map = {}

  def process(self, element, all_labels):
    all_labels = list(all_labels)
    # DataFlow cannot garuantee the order of the labels when materializing it.
    # The labels materialized and consumed by training may not be with the same order
    # as the one used in preprocessing. So we need to sort it in both preprocessing
    # and training so the order matches.
    all_labels.sort()
    if not self.label_to_id_map:
      for i, label in enumerate(all_labels):
        label = label.strip()
        if label:
          self.label_to_id_map[label] = i

    # Row format is:
    # image_uri,label_id
    if not element:
      skipped_empty_line.inc()
      return

    rows_count.inc()
    uri = element['image_url']

    try:
      label_id = self.label_to_id_map[element['label'].strip()]
    except KeyError:
      unlabeled_image.inc()
    yield uri, label_id


class ReadImageAndConvertToJpegDoFn(beam.DoFn):
  """Read files from GCS and convert images to JPEG format.

  We do this even for JPEG images to remove variations such as different number
  of channels.
  """

  def process(self, element):
    from tensorflow.python.lib.io import file_io as tf_file_io

    uri, label_id = element
    try:
      with tf_file_io.FileIO(uri, 'r') as f:
        img = Image.open(f).convert('RGB')
    # A variety of different calling libraries throw different exceptions here.
    # They all correspond to an unreadable file so we treat them equivalently.
    # pylint: disable broad-except
    except Exception as e:
      logging.exception('Error processing image %s: %s', uri, str(e))
      error_count.inc()
      return

    # Convert to desired format and output.
    output = cStringIO.StringIO()
    img.save(output, 'jpeg')
    image_bytes = output.getvalue()
    yield uri, label_id, image_bytes


class EmbeddingsGraph(object):
  """Builds a graph and uses it to extract embeddings from images.
  """

  # These constants are set by Inception v3's expectations.
  WIDTH = 299
  HEIGHT = 299
  CHANNELS = 3

  def __init__(self, tf_session, checkpoint_path):
    self.tf_session = tf_session
    # input_jpeg is the tensor that contains raw image bytes.
    # It is used to feed image bytes and obtain embeddings.
    self.input_jpeg, self.embedding = self.build_graph()
    self.tf_session.run(tf.global_variables_initializer())
    self.restore_from_checkpoint(checkpoint_path)

  def build_graph(self):
    """Forms the core by building a wrapper around the inception graph.

      Here we add the necessary input & output tensors, to decode jpegs,
      serialize embeddings, restore from checkpoint etc.

      To use other Inception models modify this file. Note that to use other
      models beside Inception, you should make sure input_shape matches
      their input. Resizing or other modifications may be necessary as well.
      See tensorflow/contrib/slim/python/slim/nets/inception_v3.py for
      details about InceptionV3.

    Returns:
      input_jpeg: A tensor containing raw image bytes as the input layer.
      embedding: The embeddings tensor, that will be materialized later.
    """

    input_jpeg = tf.placeholder(tf.string, shape=None)
    image = tf.image.decode_jpeg(input_jpeg, channels=self.CHANNELS)

    # Note resize expects a batch_size, but we are feeding a single image.
    # So we have to expand then squeeze.  Resize returns float32 in the
    # range [0, uint8_max]
    image = tf.expand_dims(image, 0)

    # convert_image_dtype also scales [0, uint8_max] -> [0 ,1).
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_bilinear(
        image, [self.HEIGHT, self.WIDTH], align_corners=False)

    # Then rescale range to [-1, 1) for Inception.
    image = tf.subtract(image, 0.5)
    inception_input = tf.multiply(image, 2.0)

    # Build Inception layers, which expect a tensor of type float from [-1, 1)
    # and shape [batch_size, height, width, channels].
    with slim.arg_scope(_inceptionlib.inception_v3_arg_scope()):
      _, end_points = _inceptionlib.inception_v3(inception_input, is_training=False)

    embedding = end_points['PreLogits']
    return input_jpeg, embedding

  def restore_from_checkpoint(self, checkpoint_path):
    """To restore inception model variables from the checkpoint file.

       Some variables might be missing in the checkpoint file, so it only
       loads the ones that are avialable, assuming the rest would be
       initialized later.
    Args:
      checkpoint_path: Path to the checkpoint file for the Inception graph.
    """
    # Get all variables to restore. Exclude Logits and AuxLogits because they
    # depend on the input data and we do not need to intialize them from
    # checkpoint.
    all_vars = tf.contrib.slim.get_variables_to_restore(
        exclude=['InceptionV3/AuxLogits', 'InceptionV3/Logits', 'global_step'])

    saver = tf.train.Saver(all_vars)
    saver.restore(self.tf_session, checkpoint_path)

  def calculate_embedding(self, batch_image_bytes):
    """Get the embeddings for a given JPEG image.

    Args:
      batch_image_bytes: As if returned from [ff.read() for ff in file_list].

    Returns:
      The Inception embeddings (bottleneck layer output)
    """
    return self.tf_session.run(
        self.embedding, feed_dict={self.input_jpeg: batch_image_bytes})


class TFExampleFromImageDoFn(beam.DoFn):
  """Embeds image bytes and labels, stores them in tensorflow.Example.

  (uri, label_ids, image_bytes) -> (tensorflow.Example).

  Output proto contains 'label', 'image_uri' and 'embedding'.
  The 'embedding' is calculated by feeding image into input layer of image
  neural network and reading output of the bottleneck layer of the network.

  Attributes:
    image_graph_uri: an uri to gcs bucket where serialized image graph is
                     stored.
  """

  def __init__(self, checkpoint_path):
    self.tf_session = None
    self.graph = None
    self.preprocess_graph = None
    self._checkpoint_path = checkpoint_path

  def start_bundle(self, context=None):
    # There is one tensorflow session per instance of TFExampleFromImageDoFn.
    # The same instance of session is re-used between bundles.
    # Session is closed by the destructor of Session object, which is called
    # when instance of TFExampleFromImageDoFn() is destructed.
    if not self.graph:
      self.graph = tf.Graph()
      self.tf_session = tf.InteractiveSession(graph=self.graph)
      with self.graph.as_default():
        self.preprocess_graph = EmbeddingsGraph(self.tf_session, self._checkpoint_path)

  def finish_bundle(self, context=None):
    if self.tf_session is not None:
      self.tf_session.close()

  def process(self, element):

    def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    uri, label_id, image_bytes = element

    try:
      embedding = self.preprocess_graph.calculate_embedding(image_bytes)
    except tf.errors.InvalidArgumentError as e:
      incompatible_image.inc()
      logging.warning('Could not encode an image from %s: %s', uri, str(e))
      return

    if embedding.any():
      embedding_good.inc()
    else:
      embedding_bad.inc()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_uri': _bytes_feature([str(uri)]),
        'embedding': _float_feature(embedding.ravel().tolist()),
    }))

    example.features.feature['label'].int64_list.value.append(label_id)

    yield example


class TrainEvalSplitPartitionFn(beam.PartitionFn):
  """Split train and eval data."""
  def partition_for(self, element, num_partitions):
    import random
    return 1 if random.random() > 0.7 else 0


class ExampleProtoCoder(beam.coders.Coder):
  """A coder to encode and decode TensorFlow Example objects."""

  def __init__(self):
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
    self._tf_train = tf.train

  def encode(self, example_proto):
    return example_proto.SerializeToString()

  def decode(self, serialized_str):
    example = self._tf_train.Example()
    example.ParseFromString(serialized_str)
    return example


class SaveFeatures(beam.PTransform):
  """Save Features in a TFRecordIO format.
  """

  def __init__(self, file_path_prefix):
    super(SaveFeatures, self).__init__('SaveFeatures')
    self._file_path_prefix = file_path_prefix

  def expand(self, features):
    return (features |
            'Write to %s' % self._file_path_prefix.replace('/', '_') >>
            tfrecordio.WriteToTFRecord(file_path_prefix=self._file_path_prefix,
                                       file_name_suffix='.tfrecord.gz',
                                       shard_name_template=fileio.DEFAULT_SHARD_NAME_TEMPLATE,
                                       coder=ExampleProtoCoder(),
                                       compression_type=fileio.CompressionTypes.AUTO))


def _labels_pipeline(sources):
  labels = (sources |
            'Flatten Sources for labels' >> beam.Flatten() |
            'Parse input for labels' >> beam.Map(lambda x: str(x['label'])) |
            'Combine labels' >> beam.transforms.combiners.Count.PerElement() |
            'Get labels' >> beam.Map(lambda label_count: label_count[0]))
  return labels


def _transformation_pipeline(source, checkpoint, labels, mode):
  transformed = (source |
                 'Extract label ids(%s)' % mode >>
                 beam.ParDo(ExtractLabelIdsDoFn(), beam.pvalue.AsIter(labels)) |
                 'Read and convert to JPEG(%s)' % mode >>
                 beam.ParDo(ReadImageAndConvertToJpegDoFn()) |
                 'Embed and make TFExample(%s)' % mode >>
                 beam.ParDo(TFExampleFromImageDoFn(checkpoint)))
  return transformed


def configure_pipeline(p, dataset_train, dataset_eval, checkpoint_path, output_dir, job_id):
  source_train = _util.get_sources_from_dataset(p, dataset_train, 'train')
  labels_source = [source_train]
  if dataset_eval is not None:
    source_eval = _util.get_sources_from_dataset(p, dataset_eval, 'eval')
    labels_source.append(source_eval)

  labels = _labels_pipeline(labels_source)
  train_preprocessed = _transformation_pipeline(source_train, checkpoint_path, labels, 'train')
  if dataset_eval is not None:
    # explicit eval data.
    eval_preprocessed = _transformation_pipeline(source_eval, checkpoint_path, labels, 'eval')
  else:
    # Split train/eval.
    train_preprocessed, eval_preprocessed = (train_preprocessed |
                                             'Random Partition' >>
                                             beam.Partition(TrainEvalSplitPartitionFn(), 2))

  output_train_path = os.path.join(output_dir, job_id, 'train')
  output_eval_path = os.path.join(output_dir, job_id, 'eval')
  labels_file = os.path.join(output_dir, job_id, 'labels')
  labels_save = (labels |
                 'Write labels' >>
                 beam.io.textio.WriteToText(labels_file, shard_name_template=''))
  train_save = train_preprocessed | 'Save train to disk' >> SaveFeatures(output_train_path)
  eval_save = eval_preprocessed | 'Save eval to disk' >> SaveFeatures(output_eval_path)
  # Make sure we write "latest" file after train and eval data are successfully written.
  output_latest_file = os.path.join(output_dir, 'latest')
  ([eval_save, train_save, labels_save] | 'Wait for train eval saving' >> beam.Flatten() |
      'Fixed One' >> beam.transforms.combiners.Sample.FixedSizeGlobally(1) |
      beam.Map(lambda path: job_id) |
      'WriteLatest' >> beam.io.textio.WriteToText(output_latest_file, shard_name_template=''))
