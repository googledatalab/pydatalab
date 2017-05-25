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

  feature_parser = parser.add_mutually_exclusive_group(required=False)
  feature_parser.add_argument('--target', dest='target', action='store_true')
  feature_parser.add_argument('--no-target', dest='target', action='store_false')
  parser.set_defaults(target=True)

  parser.add_argument(
      '--shuffle',
      action='store_true',
      default=False)

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


def preprocess(pipeline, args):
  input_metadata = metadata_io.read_metadata(
      os.path.join(args.analysis_output_dir, RAW_METADATA_DIR))

  schema = json.loads(file_io.read_file_to_string(
      os.path.join(args.analysis_output_dir, SCHEMA_FILE)).decode())
  features = json.loads(file_io.read_file_to_string(
      os.path.join(args.analysis_output_dir, FEATURES_FILE)).decode())

  column_names = [col['name'] for col in schema]

  exclude_outputs = None
  if not args.target:
    for name, transform in six.iteritems(features):
      if transform['transform'] == TARGET_TRANSFORM:
        target_name = name
        column_names.remove(target_name)
        exclude_outputs = [target_name]
        del input_metadata.schema.column_schemas[target_name]
        break

  if args.csv_file_pattern:
    coder = coders.CsvCoder(column_names, input_metadata.schema, delimiter=',')
    raw_data = (
        pipeline
        | 'ReadCsvData' >> beam.io.ReadFromText(args.csv_file_pattern)
        | 'ParseCsvData' >> beam.Map(coder.decode))
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
  raw_data = (
      raw_data
      | 'PreprocessTransferredLearningTransformations'
      >> beam.Map(prepare_image_transforms, image_columns))

  if args.shuffle:
    raw_data = raw_data | 'ShuffleData' >> shuffle()

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
