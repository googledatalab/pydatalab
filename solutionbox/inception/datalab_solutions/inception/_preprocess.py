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
from apache_beam.utils.pipeline_options import PipelineOptions
import cStringIO
import csv
import google.cloud.ml as ml
from google.cloud.ml.io import SaveFeatures
import logging
import os
from PIL import Image
import tensorflow as tf

from . import _inceptionlib
from . import _util


slim = tf.contrib.slim

error_count = beam.Aggregator('errorCount')
rows_count = beam.Aggregator('RowsCount')
skipped_empty_line = beam.Aggregator('skippedEmptyLine')
embedding_good = beam.Aggregator('embedding_good')
embedding_bad = beam.Aggregator('embedding_bad')
incompatible_image = beam.Aggregator('incompatible_image')
invalid_uri = beam.Aggregator('invalid_file_name')
ignored_unlabeled_image = beam.Aggregator('ignored_unlabeled_image')


class ExtractLabelIdsDoFn(beam.DoFn):
  """Extracts (uri, label_ids) tuples from CSV rows.
  """

  def start_bundle(self, context, *unused_args, **unused_kwargs):
    self.label_to_id_map = {}

  def process(self, context, all_labels):
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
    element = context.element
    if not element:
      context.aggregate_to(skipped_empty_line, 1)
      return

    context.aggregate_to(rows_count, 1)
    uri = element['image_url']
    if not uri or not uri.startswith('gs://'):
      context.aggregate_to(invalid_uri, 1)
      return

    try:
      label_id = self.label_to_id_map[element['label'].strip()]
    except KeyError:
      context.aggregate_to(ignored_unlabeled_image, 1)
    yield uri, label_id


class ReadImageAndConvertToJpegDoFn(beam.DoFn):
  """Read files from GCS and convert images to JPEG format.

  We do this even for JPEG images to remove variations such as different number
  of channels.
  """

  def process(self, context):
    uri, label_id = context.element

    try:
      with ml.util._file.open_local_or_gcs(uri, mode='r') as f:
        img = Image.open(f).convert('RGB')
    # A variety of different calling libraries throw different exceptions here.
    # They all correspond to an unreadable file so we treat them equivalently.
    # pylint: disable broad-except
    except Exception as e:
      logging.exception('Error processing image %s: %s', uri, str(e))
      context.aggregate_to(error_count, 1)
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
    image = tf.sub(image, 0.5)
    inception_input = tf.mul(image, 2.0)

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

  def start_bundle(self, context):
    # There is one tensorflow session per instance of TFExampleFromImageDoFn.
    # The same instance of session is re-used between bundles.
    # Session is closed by the destructor of Session object, which is called
    # when instance of TFExampleFromImageDoFn() is destructed.
    if not self.graph:
      self.graph = tf.Graph()
      self.tf_session = tf.InteractiveSession(graph=self.graph)
      with self.graph.as_default():
        self.preprocess_graph = EmbeddingsGraph(self.tf_session, self._checkpoint_path)

  def finish_bundle(self, context):
    if self.tf_session is not None:
      self.tf_session.close()

  def process(self, context):

    def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    uri, label_id, image_bytes = context.element

    try:
      embedding = self.preprocess_graph.calculate_embedding(image_bytes)
    except tf.errors.InvalidArgumentError as e:
      context.aggregate_to(incompatible_image, 1)
      logging.warning('Could not encode an image from %s: %s', uri, str(e))
      return

    if embedding.any():
      context.aggregate_to(embedding_good, 1)
    else:
      context.aggregate_to(embedding_bad, 1)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_uri': _bytes_feature([str(uri)]),
        'embedding': _float_feature(embedding.ravel().tolist()),
    }))

    example.features.feature['label'].int64_list.value.append(label_id)

    yield example


class TrainEvalSplitPartitionFn(beam.PartitionFn):
  """Split train and eval data."""
  def partition_for(self, context, num_partitions):
    import random
    return 1 if random.random() > 0.7 else 0


def _get_sources_from_csvs(p, input_paths):
  source_list = []
  for ii, input_path in enumerate(input_paths):
    source_list.append(p | 'Read from Csv %d' % ii >> 
        beam.io.ReadFromText(input_path, strip_trailing_newlines=True))
  all_sources = (source_list | 'Flatten Sources' >> beam.Flatten()
      | beam.Map(lambda line: csv.DictReader([line], fieldnames=['image_url', 'label']).next()))
  return all_sources


def _get_sources_from_bigquery(p, query):
  if len(query.split()) == 1:
    bq_source = beam.io.BigQuerySource(table=query)
  else:
    bq_source = beam.io.BigQuerySource(query=query)
  query_results = p | 'Read from BigQuery' >> beam.io.Read(bq_source)
  return query_results


def _configure_pipeline_from_source(source, checkpoint_path, output_dir, job_id):
  labels = (source
      | 'Parse input for labels' >> beam.Map(lambda x: x['label'])
      | 'Combine labels' >> beam.transforms.combiners.Count.PerElement()
      | 'Get labels' >> beam.Map(lambda label_count: label_count[0]))
  all_preprocessed = (source
      | 'Extract label ids' >> beam.ParDo(ExtractLabelIdsDoFn(),
                                          beam.pvalue.AsIter(labels))
      | 'Read and convert to JPEG' >> beam.ParDo(ReadImageAndConvertToJpegDoFn())
      | 'Embed and make TFExample' >> beam.ParDo(TFExampleFromImageDoFn(checkpoint_path)))
  train_eval = (all_preprocessed |
       'Random Partition' >> beam.Partition(TrainEvalSplitPartitionFn(), 2))
  preprocessed_train = os.path.join(output_dir, job_id, 'train')
  preprocessed_eval = os.path.join(output_dir, job_id, 'eval')
  labels_file = os.path.join(output_dir, job_id, 'labels')
  labels_save = (labels 
      | 'Write labels' >> beam.io.textio.WriteToText(labels_file, shard_name_template=''))
  eval_save = train_eval[1] | 'Save eval to disk' >> SaveFeatures(preprocessed_eval)
  train_save = train_eval[0] | 'Save train to disk' >> SaveFeatures(preprocessed_train)
  # Make sure we write "latest" file after train and eval data are successfully written.
  output_latest_file = os.path.join(output_dir, 'latest')
  ([eval_save, train_save, labels_save] | 'Wait for train eval saving' >> beam.Flatten() |
      beam.transforms.combiners.Sample.FixedSizeGlobally('Fixed One', 1) |
      beam.Map(lambda path: job_id) |
      'WriteLatest' >> beam.io.textio.WriteToText(output_latest_file, shard_name_template=''))


def configure_pipeline_csv(p, checkpoint_path, input_paths, output_dir, job_id):
  all_sources = _get_sources_from_csvs(p, input_paths)
  _configure_pipeline_from_source(all_sources, checkpoint_path, output_dir, job_id)


def configure_pipeline_bigquery(p, checkpoint_path, query, output_dir, job_id):
  all_sources = _get_sources_from_bigquery(p, query)
  _configure_pipeline_from_source(all_sources, checkpoint_path, output_dir, job_id)

