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
"""Runs prediction on a trained model."""


import argparse
import datetime
import os
import sys

import apache_beam as beam


def parse_arguments(argv):
  """Parse command line arguments.

  Args:
    argv: includes the script's name.

  Returns:
    argparse object
  """
  parser = argparse.ArgumentParser(
      description='Runs Prediction inside a beam or Dataflow job.')
  # cloud options
  parser.add_argument('--project_id',
                      help='The project to which the job will be submitted.')
  parser.add_argument('--cloud',
                      action='store_true',
                      help='Run preprocessing on the cloud.')
  parser.add_argument('--job_name',
                      default=('structured-data-batch-prediction-'
                          + datetime.datetime.now().strftime('%Y%m%d%H%M%S')),
                      help='Dataflow job name. Must be unique over all jobs.')

  # I/O args
  parser.add_argument('--predict_data',
                      required=True,
                      help='Data to run prediction on')
  parser.add_argument('--trained_model_dir',
                      required=True,
                      help='Usually train_output_path/model.')
  parser.add_argument('--output_dir',
                      required=True,
                      help=('Location to save output.'))

  # Other args
  parser.add_argument('--batch_size',
                      required=False,
                      default=1000,
                      type=int,
                      help=('Batch size. Larger values consumes more memrory '
                            'but takes less time to finish.'))
  parser.add_argument('--shard_files',
                      dest='shard_files',
                      action='store_true',
                      help='Shard files')
  parser.add_argument('--no-shard_files',
                      dest='shard_files',
                      action='store_false',
                      help='Don\'t shard files')
  parser.set_defaults(shard_files=True)
  parser.add_argument('--output_format',
                      choices=['csv', 'json'],
                      default='csv',
                      help="""
      The output results. 
        raw_json: produces a newline file where each line is json. No 
            post processing is performed and the output matches what the trained
            model produces.
        csv: produces a csv file without a header row and a header csv file.
            For classification problems, the vector of probabalities for each 
            target class is split into individual csv columns.""")

  args, _ = parser.parse_known_args(args=argv[1:])

  if args.cloud:
    if not args.project_id:
      raise ValueError('--project_id needed with --cloud')
    if not args.trained_model_dir.startswith('gs://'):
      raise ValueError('trained_model_dir needs to be a GCS path,')
    if not args.output_dir.startswith('gs://'):
      raise ValueError('output_dir needs to be a GCS path.')
    if not args.predict_data.startswith('gs://'):
      raise ValueError('predict_data needs to be a GCS path.')


  return args


class FixMissingTarget(beam.DoFn):
  """A DoFn to fix missing target columns."""

  def __init__(self, trained_model_dir):
    """Reads the schema file and extracted the expected number of columns.

    Args:
      trained_model_dir: path to model.

    Raises:
      ValueError: if schema.json not found in trained_model_dir
    """
    from tensorflow.python.lib.io import file_io
    import json
    import os

    schema_path = os.path.join(trained_model_dir, 'schema.json')
    if not file_io.file_exists(schema_path):
      raise ValueError('schema.json missing from %s' % schema_path)
    schema = json.loads(file_io.read_file_to_string(schema_path))
    self._num_expected_columns = len(schema)

  def process(self, context):
    """Fixes csv line if target is missing.

    The first column is assumed to be the target column, and the TF graph
    expects to always parse the target column, even in prediction. Below,
    we check how many csv columns there are, and if the target is missing, we
    prepend a ',' to denote the missing column. 

    Example:
      'target,key,value1,...' -> 'target,key,value1,...' (no change)
      'key,value1,...' -> ',key,value1,...' (add missing target column)

    The value of the missing target column comes from the default value given 
    to tf.decode_csv in the graph.
    """
    import logging
    import apache_beam as beam

    num_columns = len(context.element.split(','))
    if num_columns == self._num_expected_columns:
      yield context.element
    elif num_columns + 1 == self._num_expected_columns:
      yield ',' + context.element
    else:
      logging.error('Got an unexpected number of columns from [%s].' %
                    context.element)
      yield beam.pvalue.SideOutputValue('errors',
                                        ('bad columns', context.element))


class EmitAsBatchDoFn(beam.DoFn):
  """A DoFn that buffers the records and emits them batch by batch."""

  def __init__(self, batch_size):
    """Constructor of EmitAsBatchDoFn beam.DoFn class.

    Args:
      batch_size: the max size we want to buffer the records before emitting.
    """
    self._batch_size = batch_size
    self._cached = []

  def process(self, context):
    self._cached.append(context.element)
    if len(self._cached) >= self._batch_size:
      emit = self._cached
      self._cached = []
      yield emit

  def finish_bundle(self, context):
    if len(self._cached) > 0:  # pylint: disable=g-explicit-length-test
      yield self._cached


class RunGraphDoFn(beam.DoFn):
  """A DoFn for running the TF graph."""

  def __init__(self, trained_model_dir):
    self._trained_model_dir = trained_model_dir
    self._session = None

  def start_bundle(self, context=None):
    from tensorflow.contrib.session_bundle import session_bundle
    import json

    self._session, _ = session_bundle.load_session_bundle_from_path(
        self._trained_model_dir)

    # input_alias_map {'input_csv_string': tensor_name}
    self._input_alias_map = json.loads(
        self._session.graph.get_collection('inputs')[0])

    # output_alias_map {'target_from_input': tensor_name, 'key': ...}
    self._output_alias_map = json.loads(
        self._session.graph.get_collection('outputs')[0])

    self._aliases, self._tensor_names = zip(*self._output_alias_map.items())

  def finish_bundle(self, context=None):
    self._session.close()


  def process(self, context):
    import collections
    import logging
    import apache_beam as beam   

    num_in_batch = 0
    try:
      assert self._session is not None

      feed_dict = collections.defaultdict(list)
      for line in context.element:
        feed_dict[self._input_alias_map.values()[0]].append(line)
        num_in_batch += 1

      # batch_result is list of numpy arrays with batch_size many rows.
      batch_result = self._session.run(fetches=self._tensor_names,
                                       feed_dict=feed_dict)

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
            predictions[name] = (value.tolist() if getattr(value, 'tolist', None)
                                else value)
          yield predictions
      else:
        predictions = {}
        for i in range(len(self._aliases)):
          value = batch_result[i]
          value = (value.tolist() if getattr(value, 'tolist', None)
                   else value)
          predictions[self._aliases[i]] =  value
        yield predictions

    except Exception as e:  # pylint: disable=broad-except   
      logging.error('RunGraphDoFn: Bad input: %s, Error: %s',
                    str(context.element), str(e))
      yield beam.pvalue.SideOutputValue('errors',
                                        (str(e), context.element))


class RawJsonCoder(beam.coders.Coder):
  """Coder for json newline files."""

  def encode(self, obj):
    """Encodes a python object into a JSON string.

    Args:
      obj: python object.

    Returns:
      JSON string.
    """
    import json
    return json.dumps(obj, separators=(',', ': '))


class CSVCoder(beam.coders.Coder):
  """Coder for CSV files containing the ouput of prediction."""

  def __init__(self, header):
    """Sets the headers in the csv file.

    Args:
      header: list of strings that correspond to keys in the predictions dict.
    """
    self._header = header

  def make_header_string(self):
    return ','.join(self._header)

  def encode(self, tf_graph_predictions):
    """Encodes the graph json prediction into csv.

    Args:
      tf_graph_predictions: python dict.

    Returns:
      csv string.
    """
    row = []
    for col in self._header:
      row.append(str(tf_graph_predictions[col]))

    return ','.join(row)


class FormatAndSave(beam.PTransform):

  def __init__(self, args):
    self._shard_name_template = None if args.shard_files else ''
    self._output_format = args.output_format
    self._output_dir = args.output_dir

    # See if the target vocab should be loaded.
    if self._output_format == 'csv':
      from tensorflow.contrib.session_bundle import session_bundle
      import json

      self._session, _ = session_bundle.load_session_bundle_from_path(
          args.trained_model_dir)
     
      # output_alias_map {'target_from_input': tensor_name, 'key': ...}
      output_alias_map = json.loads(
          self._session.graph.get_collection('outputs')[0])

      self._header = sorted(output_alias_map.keys())
      self._session.close()


  def apply(self, datasets):
    return self.expand(datasets)

  def expand(self, datasets):
    tf_graph_predictions, errors = datasets

    if self._output_format == 'json':
      _ = (
          tf_graph_predictions 
          | 'Write Raw JSON'
          >> beam.io.textio.WriteToText(
              os.path.join(self._output_dir, 'predictions'),
              file_name_suffix='.json',
              coder=RawJsonCoder(),
              shard_name_template=self._shard_name_template))
    elif self._output_format == 'csv':
      # make a csv header file 
      csv_coder = CSVCoder(self._header)
      _ = (
          tf_graph_predictions.pipeline
          | 'Make CSV Header'
          >> beam.Create([csv_coder.make_header_string()])
          | 'Write CSV Header File'
          >> beam.io.textio.WriteToText(
              os.path.join(self._output_dir, 'csv_header'),
              file_name_suffix='.txt',
              shard_name_template=''))   
     
      # Write the csv predictions
      _ = (
          tf_graph_predictions 
          | 'Write CSV'
          >> beam.io.textio.WriteToText(
              os.path.join(self._output_dir, 'predictions'),
              file_name_suffix='.csv',
              coder=csv_coder,
              shard_name_template=self._shard_name_template))
    else:
      raise ValueError('FormatAndSave: unknown format %s', self._output_format)


    # Write the errors to a text file.
    _ = (errors
         | 'Write Errors'
         >> beam.io.textio.WriteToText(
             os.path.join(self._output_dir, 'errors'),
             file_name_suffix='.txt',
             shard_name_template=self._shard_name_template))    


def make_prediction_pipeline(pipeline, args):
  """Builds the prediction pipeline.

  Reads the csv files, prepends a ',' if the target column is missing, run 
  prediction, and then prints the formated results to a file.

  Args:
    pipeline: the pipeline
    args: command line args
  """
  

  predicted_values, errors = (
      pipeline
      | 'Read CSV Files'
      >> beam.io.ReadFromText(args.predict_data,
                              strip_trailing_newlines=True)
      | 'Is Target Missing'
      >> beam.ParDo(FixMissingTarget(args.trained_model_dir))
      | 'Batch Input'
      >> beam.ParDo(EmitAsBatchDoFn(args.batch_size))
      | 'Run TF Graph on Batches'
      >> (beam.ParDo(RunGraphDoFn(args.trained_model_dir))
          .with_outputs('errors', main='main')))

  _ = (
    (predicted_values, errors)
    | 'Format and Save'
    >> FormatAndSave(args))


def main(argv=None):
  args = parse_arguments(sys.argv if argv is None else argv)

  if args.cloud:
    options = {
        'staging_location': os.path.join(args.output_dir, 'tmp', 'staging'),
        'temp_location': os.path.join(args.output_dir, 'tmp', 'staging'),
        'job_name': args.job_name, 
        'project': args.project_id,
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    # Or use BlockingDataflowPipelineRunner
    p = beam.Pipeline('DataflowPipelineRunner', options=opts)
  else:
    p = beam.Pipeline('DirectPipelineRunner')

  make_prediction_pipeline(p, args)

  if args.cloud:
    print(('Dataflow Job submitted, see Job %s at '
           'https://console.developers.google.com/dataflow?project=%s') %
          (options['job_name'], args.project_id))
    sys.stdout.flush()

  p.run()


if __name__ == '__main__':
  main()
