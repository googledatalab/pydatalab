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
import datetime
import json
import logging
import os
import sys
import apache_beam as beam
import textwrap


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

  source_group = parser.add_mutually_exclusive_group(required=True)

  source_group.add_argument(
      '--csv',
      metavar='FILE',
      required=False,
      action='append',
      help='CSV data to transform.')

  source_group.add_argument(
      '--bigquery',
      metavar='PROJECT_ID.DATASET.TABLE_NAME',
      type=str,
      required=False,
      help=('Must be in the form `project.dataset.table_name`. BigQuery '
            'data to transform'))

  parser.add_argument(
      '--analysis',
      metavar='ANALYSIS_OUTPUT_DIR',
      required=True,
      help='The output folder of analyze')

  parser.add_argument(
      '--prefix',
      metavar='OUTPUT_FILENAME_PREFIX',
      required=True,
      type=str)

  parser.add_argument(
      '--output',
      metavar='DIR',
      default=None,
      required=True,
      help=('Google Cloud Storage or Local directory in which '
            'to place outputs.'))

  parser.add_argument(
      '--shuffle',
      action='store_true',
      default=False,
      help='If used, data source is shuffled. This is recommended for training data.')

  parser.add_argument(
      '--batch-size',
      metavar='N',
      type=int,
      default=100,
      help='Larger values increase performance and peak memory usage.')

  cloud_group = parser.add_argument_group(
      title='Cloud Parameters',
      description='These parameters are only used if --cloud is used.')

  cloud_group.add_argument(
      '--cloud',
      action='store_true',
      help='Run preprocessing on the cloud.')

  cloud_group.add_argument(
      '--job-name',
      type=str,
      help='Unique dataflow job name.')

  cloud_group.add_argument(
      '--project-id',
      help='The project to which the job will be submitted.')

  cloud_group.add_argument(
      '--num-workers',
      metavar='N',
      type=int,
      default=0,
      help='Set to 0 to use the default size determined by the Dataflow service.')

  cloud_group.add_argument(
      '--worker-machine-type',
      metavar='NAME',
      type=str,
      help='A machine name from https://cloud.google.com/compute/docs/machine-types. '
           ' If not given, the service uses the default machine type.')

  cloud_group.add_argument(
      '--async',
      action='store_true',
      help='If used, this script returns before the dataflow job is completed.')

  args = parser.parse_args(args=argv[1:])

  if args.cloud and not args.project_id:
    raise ValueError('--project-id is needed for --cloud')

  if args.async and not args.cloud:
    raise ValueError('--async should only be used with --cloud')

  if not args.job_name:
    args.job_name = ('dataflow-job-{}'.format(
        datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
  return args


@beam.ptransform_fn
def shuffle(pcoll):  # pylint: disable=invalid-name
  import random
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
  import six
  from trainer import feature_transforms

  img_cols = []
  for name, transform in six.iteritems(features):
    if transform['transform'] == feature_transforms.IMAGE_TRANSFORM:
      img_cols.append(name)

  return img_cols


def prepare_image_transforms(element, image_columns):
  """Replace an images url with its jpeg bytes.

  Args: 
    element: one input row, as a dict
    image_columns: list of columns that are image paths

  Return:
    element, where each image file path has been replaced by a base64 image.
  """
  import base64
  import cStringIO
  from PIL import Image
  from tensorflow.python.lib.io import file_io as tf_file_io
  from apache_beam.metrics import Metrics

  img_error_count = Metrics.counter('main', 'ImgErrorCount')
  img_missing_count = Metrics.counter('main', 'ImgMissingCount')

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
    from apache_beam.transforms import window
    from apache_beam.utils.windowed_value import WindowedValue

    if len(self._cached) > 0:  # pylint: disable=g-explicit-length-test
      yield WindowedValue(self._cached, -1, [window.GlobalWindow()])


class TransformFeaturesDoFn(beam.DoFn):
  """Converts raw data into transformed data."""

  def __init__(self, analysis_output_dir, features, schema, stats):
    self._analysis_output_dir = analysis_output_dir
    self._features = features
    self._schema = schema
    self._stats = stats
    self._session = None

  def start_bundle(self, element=None):
    """Build the transfromation graph once."""
    import tensorflow as tf
    from trainer import feature_transforms

    g = tf.Graph()
    session = tf.Session(graph=g)

    # Build the transformation graph
    with g.as_default():
      transformed_features, _, placeholders = (
          feature_transforms.build_csv_serving_tensors_for_transform_step(
              analysis_path=self._analysis_output_dir, 
              features=self._features, 
              schema=self._schema,
              stats=self._stats,
              keep_target=True))
      session.run(tf.tables_initializer())
    
    self._session = session
    self._transformed_features = transformed_features
    self._input_placeholder_tensor = placeholders['csv_example']

  def finish_bundle(self, element=None):
    self._session.close()

  def process(self, element):
    """Run the transformation graph on batched input data

    Args:
      element: list of csv strings, representing one batch input to the TF graph.

    Returns:
      dict containing the transformed data. Results are un-batched. Sparse
      tensors are converted to lists.
    """
    import apache_beam as beam
    import six
    import tensorflow as tf

    try:
      clean_element = []
      for line in element:
        clean_element.append(line.rstrip())

      # batch_result is list of numpy arrays with batch_size many rows.
      batch_result = self._session.run(
          fetches=self._transformed_features,
          feed_dict={self._input_placeholder_tensor: clean_element})

      # ex batch_result. 
      # Dense tensor: {'col1': array([[batch_1], [batch_2]])}
      # Sparse tensor: {'col1': tf.SparseTensorValue(
      #   indices=array([[batch_1, 0], [batch_1, 1], ...,
      #                  [batch_2, 0], [batch_2, 1], ...]],
      #   values=array[value, value, value, ...])}

      # Unbatch the results.
      for i in range(len(clean_element)):
        transformed_features = {}
        for name, value in six.iteritems(batch_result):
          if isinstance(value, tf.SparseTensorValue):
            batch_i_indices = value.indices[:, 0] == i
            batch_i_values = value.values[batch_i_indices]
            transformed_features[name] = batch_i_values.tolist()
          else:
            transformed_features[name] = value[i].tolist()

        yield transformed_features

    except Exception as e:  # pylint: disable=broad-except
      yield beam.pvalue.SideOutputValue('errors',
                                        (str(e), element))


def decode_csv(csv_string, column_names):
  """Parse a csv line into a dict.

  Args:
    csv_string: a csv string. May contain missing values "a,,c"
    column_names: list of column names

  Returns:
    Dict of {column_name, value_from_csv}. If there are missing values, 
    value_from_csv will be ''.
  """
  import csv
  r = next(csv.reader([csv_string]))
  if len(r) != len(column_names):
    raise ValueError('csv line %s does not have %d columns' % (csv_string, len(column_names)))
  return {k: v for k, v in zip(column_names, r)}


def encode_csv(data_dict, column_names):
  """Builds a csv string.

  Args:
    data_dict: dict of {column_name: 1 value}
    column_names: list of column names

  Returns:
    A csv string version of data_dict
  """
  import csv
  import six
  values = [str(data_dict[x]) for x in column_names]
  str_buff = six.StringIO()
  writer = csv.writer(str_buff, lineterminator='')
  writer.writerow(values)
  return str_buff.getvalue()


def serialize_example(transformed_json_data, info_dict):
  """Makes a serialized tf.example.

  Args:
    transformed_json_data: dict of transformed data.
    info_dict: output of feature_transforms.get_transfrormed_feature_info()

  Returns:
    The serialized tf.example version of transformed_json_data.
  """
  import six
  import tensorflow as tf

  def _make_int64_list(x):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=x))
  def _make_bytes_list(x):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=x))
  def _make_float_list(x):
    return tf.train.Feature(float_list=tf.train.FloatList(value=x))

  if sorted(six.iterkeys(transformed_json_data)) != sorted(six.iterkeys(info_dict)):
    raise ValueError('Keys do not match %s, %s' % (six.iterkeys(transformed_json_data), six.iterkeys(info_dict)))

  ex_dict = {}
  for name, info in six.iteritems(info_dict):
    if info['dtype'] == tf.int64:
      ex_dict[name] = _make_int64_list(transformed_json_data[name])
    elif info['dtype'] == tf.float32:
      ex_dict[name] = _make_float_list(transformed_json_data[name])
    elif info['dtype'] == tf.string:
      ex_dict[name] = _make_bytes_list(transformed_json_data[name])      
    else:
      raise ValueError('Unsupported data type %s' % info['dtype'])

  ex = tf.train.Example(features=tf.train.Features(feature=ex_dict))
  return ex.SerializeToString()


def preprocess(pipeline, args):
  """Transfrom csv data into transfromed tf.example files.

  Outline:
    1) read the input data (as csv or bigquery) into a dict format
    2) replace image paths with base64 encoded image files
    3) build a csv input string with images paths replaced with base64. This 
       matches the serving csv that a trained model would expect.
    4) batch the csv strings
    5) run the transformations
    6) write the results to tf.example files and save any errors.
  """
  from tensorflow.python.lib.io import file_io
  from trainer import feature_transforms

  schema = json.loads(file_io.read_file_to_string(
      os.path.join(args.analysis, feature_transforms.SCHEMA_FILE)).decode())
  features = json.loads(file_io.read_file_to_string(
      os.path.join(args.analysis, feature_transforms.FEATURES_FILE)).decode())
  stats = json.loads(file_io.read_file_to_string(
      os.path.join(args.analysis, feature_transforms.STATS_FILE)).decode())

  column_names = [col['name'] for col in schema]

  if args.csv:
    all_files = []
    for i, file_pattern in enumerate(args.csv):
      all_files.append(pipeline | ('ReadCSVFile%d' % i) >> beam.io.ReadFromText(file_pattern))
    raw_data = (
        all_files
        | 'MergeCSVFiles' >> beam.Flatten()
        | 'ParseCSVData' >> beam.Map(decode_csv, column_names))
  else:
    columns = ', '.join(column_names)
    query = 'SELECT {columns} FROM `{table}`'.format(columns=columns,
                                                     table=args.bigquery)
    raw_data = (
        pipeline
        | 'ReadBiqQueryData'
        >> beam.io.Read(beam.io.BigQuerySource(query=query,
                                               use_standard_sql=True)))

  # Note that prepare_image_transforms does not make embeddings, it justs reads
  # the image files and converts them to byte stings. TransformFeaturesDoFn()
  # will make the image embeddings.
  image_columns = image_transform_columns(features)
  
  clean_csv_data = (
      raw_data
      | 'PreprocessTransferredLearningTransformations'
      >> beam.Map(prepare_image_transforms, image_columns)
      | 'BuildCSVString'
      >> beam.Map(encode_csv, column_names))

  if args.shuffle:
    clean_csv_data = clean_csv_data | 'ShuffleData' >> shuffle()

  transform_dofn = TransformFeaturesDoFn(args.analysis, features, schema, stats)
  (transformed_data, errors) = (
       clean_csv_data
       | 'Batch Input' 
       >> beam.ParDo(EmitAsBatchDoFn(args.batch_size)) 
       | 'Run TF Graph on Batches' 
       >> beam.ParDo(transform_dofn).with_outputs('errors', main='main'))

  _ = (transformed_data
        | 'SerializeExamples' >> beam.Map(serialize_example, feature_transforms.get_transformed_feature_info(features, schema))
        | 'WriteExamples'
        >> beam.io.WriteToTFRecord(
            os.path.join(args.output, args.prefix),
            file_name_suffix='.tfrecord.gz'))
  _ = (errors
       | 'WriteErrors'
       >> beam.io.WriteToText(
           os.path.join(args.output, 'errors_' + args.prefix),
           file_name_suffix='.txt'))


def main(argv=None):
  """Run Preprocessing as a Dataflow."""
  args = parse_arguments(sys.argv if argv is None else argv)
  temp_dir = os.path.join(args.output, 'tmp')

  if args.cloud:
    pipeline_name = 'DataflowRunner'
  else:
    pipeline_name = 'DirectRunner'
    # Suppress TF cpp warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

  options = {
      'job_name': args.job_name,
      'temp_location': temp_dir,
      'project': args.project_id,
      'setup_file':
          os.path.abspath(os.path.join(
              os.path.dirname(__file__),
              'setup.py')),
  }
  if args.num_workers:
    options['num_workers'] = args.num_workers
  if args.worker_machine_type:
    options['worker_machine_type'] = args.worker_machine_type

  pipeline_options = beam.pipeline.PipelineOptions(flags=[], **options)

  p = beam.Pipeline(pipeline_name, options=pipeline_options)
  preprocess(pipeline=p, args=args)
  pipeline_result = p.run()

  if not args.async:
    pipeline_result.wait_until_finish()
  if args.async and args.cloud:
    print('View job at https://console.developers.google.com/dataflow/job/%s?project=%s' %
        (pipeline_result.job_id(), args.project_id))


if __name__ == '__main__':
  main()
