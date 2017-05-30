# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Flake8 cannot disable a warning for the file. Flake8 does not like beam code
# and reports many 'W503 line break before binary operator' errors. So turn off
# flake8 for this file.
# flake8: noqa
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import base64
import datetime
import json
import logging
import os
import random
import sys
import apache_beam as beam
from apache_beam.metrics import Metrics
import six
import textwrap
from tensorflow.python.lib.io import file_io
from tensorflow_transform import coders
from tensorflow_transform.beam import impl as tft
from tensorflow_transform.beam import tft_beam_io
from tensorflow_transform.tf_metadata import metadata_io

import trainer

img_error_count = Metrics.counter('main', 'ImgErrorCount')
img_missing_count = Metrics.counter('main', 'ImgMissingCount')

# Files
SCHEMA_FILE = 'schema.json'
FEATURES_FILE = 'features.json'

TRANSFORMED_METADATA_DIR = 'transformed_metadata'
RAW_METADATA_DIR = 'raw_metadata'
TRANSFORM_FN_DIR = 'transform_fn'

# Individual transforms
TARGET_TRANSFORM = 'target'
IMAGE_TRANSFORM = 'image_to_vec'


def parse_arguments(argv):
  """Parse command line arguments.
  Args:
    argv: list of command line arguments including program name.
  Returns:
    The parsed arguments as returned by argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser(
      formatter_class=argparse.RawDescriptionHelpFormatter,
      description=textwrap.dedent("""\
          Runs preprocessing on raw data for TensorFlow training.

          This script applies some transformations to raw data to improve
          training performance. Some data transformations can be expensive
          such as the tf-idf text column transformation. During training, the
          same raw data row might be used multiply times to train a model. This
          means the same transformations are applied to the same data row
          multiple times. This can be very inefficient, so this script applies
          partial transformations to the raw data and writes an intermediate 
          preprocessed datasource to disk for training. 

          Running this transformation step is required for two usage paths:
            1) If the img_url_to_vec transform is used. This is because
               preprocessing as image is expensive and TensorFlow cannot easily
               read raw image files during training.
            2) If the raw data is in BigQuery. TensorFlow cannot read from a 
               BigQuery source.

          Running this transformation step is recommended if a text transform is
          used (like tf-idf or bag-of-words), and the text value for each row
          is very long.

          Running this transformation step may not have an interesting training
          performance impact if the transforms are all simple like scaling
          numerical values."""))

  parser.add_argument(
      '--project-id',
      help='The project to which the job will be submitted. Only needed if '
           '--cloud is used.')
  parser.add_argument(
      '--cloud',
      action='store_true',
      help='Run preprocessing on the cloud.')
  parser.add_argument(
      '--job-name',
      type=str,
      help='Unique job name if running on the cloud.')

  parser.add_argument(
      '--csv-file-pattern',
      required=False,
      help='CSV data to transform.')
  # If using bigquery table
  parser.add_argument(
      '--bigquery-table',
      type=str,
      required=False,
      help=('Must be in the form `project.dataset.table_name`. BigQuery '
            'data to transform'))

  parser.add_argument(
      '--analysis-output-dir',
      required=True,
      help='The output folder of analyze')
  parser.add_argument(
      '--output-filename-prefix',
      required=True,
      type=str)
  parser.add_argument(
      '--output-dir',
      default=None,
      required=True,
      help=('Google Cloud Storage or Local directory in which '
            'to place outputs.'))

  parser.add_argument(
      '--shuffle',
      action='store_true',
      default=True)

  args, _ = parser.parse_known_args(args=argv[1:])

  if args.cloud and not args.project_id:
    raise ValueError('--project-id is needed for --cloud')

  if not args.job_name:
    args.job_name = ('dataflow-job-{}'.format(
        datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
  return args


@beam.ptransform_fn
def shuffle(pcoll):  # pylint: disable=invalid-name
  return (pcoll
          | 'PairWithRandom' >> beam.Map(lambda x: (random.random(), x))
          | 'GroupByRandom' >> beam.GroupByKey()
          | 'DropRandom' >> beam.FlatMap(lambda (k, vs): vs))


def image_transform_columns(features):
  """Returns a list of columns that prepare_image_transforms() should run on.

  Because of beam + pickle, IMAGE_URL_TO_VEC_TRANSFORM cannot be used inside of
  a beam function, so we extract the columns prepare_image_transforms() should 
  run on outside of beam.
  """
  img_cols = []
  for name, transform in six.iteritems(features):
    if transform['transform'] == IMAGE_TRANSFORM:
      img_cols.append(name)

  return img_cols


def prepare_image_transforms(element, image_columns):
  """Replace an images url with its jpeg bytes.

  Args: 
  """
  import cStringIO
  from PIL import Image
  from tensorflow.python.lib.io import file_io as tf_file_io

  for name in image_columns:
    uri = element[name]
    if not uri:
      img_missing_count.inc()
      continue
    try:
      with tf_file_io.FileIO(uri, 'r') as f:
        img = Image.open(f).convert('RGB')

    # A variety of different calling libraries throw different exceptions here.
    # They all correspond to an unreadable file so we treat them equivalently.
    # pylint: disable broad-except
    except Exception as e:
      logging.exception('Error processing image %s: %s', uri, str(e))
      img_error_count.inc()
      return

    # Convert to desired format and output.
    output = cStringIO.StringIO()
    img.save(output, 'jpeg')
    element[name] = base64.urlsafe_b64encode(output.getvalue())

  return element

class EmitAsBatchDoFn(beam.DoFn):
  """A DoFn that buffers the records and emits them batch by batch."""

  def __init__(self, batch_size):
    """Constructor of EmitAsBatchDoFn beam.DoFn class.

    Args:
      batch_size: the max size we want to buffer the records before emitting.
    """
    self._batch_size = batch_size
    self._cached = []

  def process(self, element):
    self._cached.append(element)
    if len(self._cached) >= self._batch_size:
      emit = self._cached
      self._cached = []
      yield emit

  def finish_bundle(self, element=None):
    if len(self._cached) > 0:  # pylint: disable=g-explicit-length-test
      yield self._cached

class TransformFeaturesDoFn(beam.DoFn):
  """TODO"""

  def __init__(self, analysis_output_dir, features, schema):
    self._analysis_output_dir = analysis_output_dir
    self._session = None

  def start_bundle(self, element=None):
    import tensorflow as tf
    g = tf.Graph().as_default()
    self._session = tf.Session(graph=g)
    transformed_features, _, placeholders = trainer.build_csv_serving_tensors(analysis_output_dir, features, schema, keep_target=True)

    self._transformed_features = transformed_features
    self._input_placeholder_tensor = placeholders['csv_example']


  def finish_bundle(self, element=None):
    self._session.close()

  def process(self, element):
    """Run batch prediciton on a TF graph.

    Args:
      element: list of csv strings, representing one batch input to the TF graph.
    """
    import collections
    import apache_beam as beam

    num_in_batch = 0
    try:
      assert self._session is not None

      feed_dict = collections.defaultdict(list)
      clean_element = []
      for line in element:
        clean_element.append(line.rstrip())

      # batch_result is list of numpy arrays with batch_size many rows.
      batch_result = self._session.run(
          fetches=self._transformed_features,
          feed_dict={self._input_placeholder_tensor: clean_element})

      # ex batch_result for batch_size > 1:
      # (array([value1, value2, ..., value_batch_size]),
      #  array([[a1, b1, c1]], ..., [a_batch_size, b_batch_size, c_batch_size]]),
      #  ...)
      # ex batch_result for batch_size == 1:
      # (value,
      #  array([a1, b1, c1]),
      #  ...)

      # Convert the results into a dict and unbatch the results.
      if num_in_batch > 1:
        for result in zip(*batch_result):
          predictions = {}
          for name, value in zip(self._aliases, result):
            predictions[name] = (value.tolist() if getattr(value, 'tolist', None) else value)
          yield predictions
      else:
        predictions = {}
        for i in range(len(self._aliases)):
          value = batch_result[i]
          value = (value.tolist() if getattr(value, 'tolist', None)
                   else value)
          predictions[self._aliases[i]] = value
        yield predictions

    except Exception as e:  # pylint: disable=broad-except
      yield beam.pvalue.SideOutputValue('errors',
                                        (str(e), element))

def decode_csv(csv_string, column_names):
  import csv
  r = next(csv.reader([csv_string]))
  if len(r) != len(column_names):
    raise ValueError('csv line %s does not have %d columns' % (csv_string, len(column_names)))
  return {k: v for k, v in zip(column_names, r)}

def encode_csv(data_dict, column_names):
  values = [str(data_dict[x]) for x in column_names]
  return ','.join(values)

def preprocess(pipeline, args):
  schema = json.loads(file_io.read_file_to_string(
      os.path.join(args.analysis_output_dir, SCHEMA_FILE)).decode())
  features = json.loads(file_io.read_file_to_string(
      os.path.join(args.analysis_output_dir, FEATURES_FILE)).decode())

  column_names = [col['name'] for col in schema]

  if args.csv_file_pattern:
    coder = coders.CsvCoder(column_names, input_metadata.schema, delimiter=',')
    raw_data = (
        pipeline
        | 'ReadCsvData' >> beam.io.ReadFromText(args.csv_file_pattern)
        | 'ParseCsvData' >> beam.Map(decode_csv, column_names))
  else:
    columns = ', '.join(column_names)
    query = 'SELECT {columns} FROM `{table}`'.format(columns=columns,
                                                     table=args.bigquery_table)
    raw_data = (
        pipeline
        | 'ReadBiqQueryData'
        >> beam.io.Read(beam.io.BigQuerySource(query=query,
                                               use_standard_sql=True)))

  # Note that prepare_image_transforms does not make embeddings, it justs reads
  # the image files and converts them to byte stings. tft.TransformDataset()
  # will apply the saved model that makes the image embeddings.
  image_columns = image_transform_columns(features)
  
  clean_csv_data = (
      raw_data
      | 'PreprocessTransferredLearningTransformations'
      >> beam.Map(prepare_image_transforms, image_columns)
      | 'BuildCSVString'
      >> beam.Map(encode_csv, column_names)))

  if args.shuffle:
    clean_csv_data = clean_csv_data | 'ShuffleData' >> shuffle()

  transform_fn = (
      pipeline
      | 'ReadTransformFn'
      >> tft_beam_io.ReadTransformFn(args.analysis_output_dir))

  (transformed_data, transform_metadata) = (
      ((raw_data, input_metadata), transform_fn)
      | 'ApplyTensorflowPreprocessingGraph' 
      >> tft.TransformDataset(exclude_outputs))

  tfexample_coder = coders.ExampleProtoCoder(transform_metadata.schema)
  _ = (transformed_data
       | 'SerializeExamples' >> beam.Map(tfexample_coder.encode)
       | 'WriteExamples'
       >> beam.io.WriteToTFRecord(
           os.path.join(args.output_dir, args.output_filename_prefix),
           file_name_suffix='.tfrecord.gz'))


def main(argv=None):
  """Run Preprocessing as a Dataflow."""
  args = parse_arguments(sys.argv if argv is None else argv)
  temp_dir = os.path.join(args.output_dir, 'tmp')

  if args.cloud:
    pipeline_name = 'DataflowRunner'
  else:
    pipeline_name = 'DirectRunner'

  options = {
      'job_name': args.job_name,
      'temp_location': temp_dir,
      'project': args.project_id,
      'setup_file':
          os.path.abspath(os.path.join(
              os.path.dirname(__file__),
              'setup.py')),
  }
  pipeline_options = beam.pipeline.PipelineOptions(flags=[], **options)

  with beam.Pipeline(pipeline_name, options=pipeline_options) as p:
    with tft.Context(temp_dir=temp_dir):
      preprocess(
          pipeline=p,
          args=args)


if __name__ == '__main__':
  main()
