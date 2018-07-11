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
import shutil
import sys
import tempfile
from tensorflow.python.lib.io import file_io

import apache_beam as beam
from apache_beam.transforms import window
from apache_beam.utils.windowed_value import WindowedValue


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
  parser.add_argument('--project-id',
                      help='The project to which the job will be submitted.')
  parser.add_argument('--cloud',
                      action='store_true',
                      help='Run preprocessing on the cloud.')
  parser.add_argument('--job-name',
                      default=('mltoolbox-batch-prediction-' +
                               datetime.datetime.now().strftime('%Y%m%d%H%M%S')),
                      help='Dataflow job name. Must be unique over all jobs.')
  parser.add_argument('--extra-package',
                      default=[],
                      action='append',
                      help=('If using --cloud, also installs these packages on '
                            'each dataflow worker'))

  # I/O args
  parser.add_argument('--predict-data',
                      required=True,
                      help='Data to run prediction on')
  parser.add_argument('--trained-model-dir',
                      required=True,
                      help='Usually train_output_path/model.')
  parser.add_argument('--output-dir',
                      required=True,
                      help=('Location to save output.'))

  # Other args
  parser.add_argument('--batch-size',
                      required=False,
                      default=1000,
                      type=int,
                      help=('Batch size. Larger values consumes more memrory '
                            'but takes less time to finish.'))
  parser.add_argument('--shard-files',
                      dest='shard_files',
                      action='store_true',
                      help='Shard files')
  parser.add_argument('--no-shard-files',
                      dest='shard_files',
                      action='store_false',
                      help='Don\'t shard files')
  parser.set_defaults(shard_files=True)

  parser.add_argument('--output-format',
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
      raise ValueError('--project-id needed with --cloud')
    if not args.trained_model_dir.startswith('gs://'):
      raise ValueError('--trained-model-dir needs to be a GCS path,')
    if not args.output_dir.startswith('gs://'):
      raise ValueError('--output-dir needs to be a GCS path.')
    if not args.predict_data.startswith('gs://'):
      raise ValueError('--predict-data needs to be a GCS path.')

  return args


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
      yield WindowedValue(self._cached, -1, [window.GlobalWindow()])


class RunGraphDoFn(beam.DoFn):
  """A DoFn for running the TF graph."""

  def __init__(self, trained_model_dir):
    self._trained_model_dir = trained_model_dir
    self._session = None

  def start_bundle(self, element=None):
    from tensorflow.python.saved_model import tag_constants
    from tensorflow.contrib.session_bundle import bundle_shim

    self._session, meta_graph = bundle_shim.load_session_bundle_or_saved_model_bundle_from_path(
        self._trained_model_dir, tags=[tag_constants.SERVING])
    signature = meta_graph.signature_def['serving_default']

    # get the mappings between aliases and tensor names
    # for both inputs and outputs
    self._input_alias_map = {friendly_name: tensor_info_proto.name
                             for (friendly_name, tensor_info_proto) in signature.inputs.items()}
    self._output_alias_map = {friendly_name: tensor_info_proto.name
                              for (friendly_name, tensor_info_proto) in signature.outputs.items()}
    self._aliases, self._tensor_names = zip(*self._output_alias_map.items())

  def finish_bundle(self, element=None):
    import tensorflow as tf

    self._session.close()
    tf.reset_default_graph()

  def process(self, element):
    """Run batch prediciton on a TF graph.

    Args:
      element: list of strings, representing one batch input to the TF graph.
    """
    import collections
    import apache_beam as beam

    num_in_batch = 0
    try:
      assert self._session is not None

      feed_dict = collections.defaultdict(list)
      for line in element:

        # Remove trailing newline.
        if line.endswith('\n'):
          line = line[:-1]

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
      yield beam.pvalue.TaggedOutput('errors', (str(e), element))


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
  """Coder for CSV files containing the output of prediction."""

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

    # Get the BQ schema if csv.
    if self._output_format == 'csv':
      from tensorflow.python.saved_model import tag_constants
      from tensorflow.contrib.session_bundle import bundle_shim
      from tensorflow.core.framework import types_pb2

      session, meta_graph = bundle_shim.load_session_bundle_or_saved_model_bundle_from_path(
          args.trained_model_dir, tags=[tag_constants.SERVING])
      signature = meta_graph.signature_def['serving_default']

      self._schema = []
      for friendly_name in sorted(signature.outputs):
        tensor_info_proto = signature.outputs[friendly_name]

        # TODO(brandondutra): Could dtype be DT_INVALID?
        # Consider getting the dtype from the graph via
        # session.graph.get_tensor_by_name(tensor_info_proto.name).dtype)
        dtype = tensor_info_proto.dtype
        if dtype == types_pb2.DT_FLOAT or dtype == types_pb2.DT_DOUBLE:
          bq_type = 'FLOAT'
        elif dtype == types_pb2.DT_INT32 or dtype == types_pb2.DT_INT64:
          bq_type = 'INTEGER'
        else:
          bq_type = 'STRING'

        self._schema.append({'mode': 'NULLABLE',
                             'name': friendly_name,
                             'type': bq_type})
      session.close()

  def apply(self, datasets):
    return self.expand(datasets)

  def expand(self, datasets):
    import json

    tf_graph_predictions, errors = datasets

    if self._output_format == 'json':
      (tf_graph_predictions |
       'Write Raw JSON' >>
       beam.io.textio.WriteToText(os.path.join(self._output_dir, 'predictions'),
                                  file_name_suffix='.json',
                                  coder=RawJsonCoder(),
                                  shard_name_template=self._shard_name_template))
    elif self._output_format == 'csv':
      # make a csv header file
      header = [col['name'] for col in self._schema]
      csv_coder = CSVCoder(header)
      (tf_graph_predictions.pipeline |
       'Make CSV Header' >>
       beam.Create([json.dumps(self._schema, indent=2)]) |
       'Write CSV Schema File' >>
       beam.io.textio.WriteToText(os.path.join(self._output_dir, 'csv_schema'),
                                  file_name_suffix='.json',
                                  shard_name_template=''))

      # Write the csv predictions
      (tf_graph_predictions |
       'Write CSV' >>
       beam.io.textio.WriteToText(os.path.join(self._output_dir, 'predictions'),
                                  file_name_suffix='.csv',
                                  coder=csv_coder,
                                  shard_name_template=self._shard_name_template))
    else:
      raise ValueError('FormatAndSave: unknown format %s', self._output_format)

    # Write the errors to a text file.
    (errors |
     'Write Errors' >>
     beam.io.textio.WriteToText(os.path.join(self._output_dir, 'errors'),
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

  # DF bug: DF does not work with unicode strings
  predicted_values, errors = (
      pipeline |
      'Read CSV Files' >>
      beam.io.ReadFromText(str(args.predict_data),
                           strip_trailing_newlines=True) |
      'Batch Input' >>
      beam.ParDo(EmitAsBatchDoFn(args.batch_size)) |
      'Run TF Graph on Batches' >>
      beam.ParDo(RunGraphDoFn(args.trained_model_dir)).with_outputs('errors', main='main'))

  ((predicted_values, errors) |
   'Format and Save' >>
   FormatAndSave(args))


def main(argv=None):
  args = parse_arguments(sys.argv if argv is None else argv)

  if args.cloud:
    tmpdir = tempfile.mkdtemp()
    try:
      local_packages = [os.path.join(tmpdir, os.path.basename(p)) for p in args.extra_package]
      for source, dest in zip(args.extra_package, local_packages):
        file_io.copy(source, dest, overwrite=True)

      options = {
          'staging_location': os.path.join(args.output_dir, 'tmp', 'staging'),
          'temp_location': os.path.join(args.output_dir, 'tmp', 'staging'),
          'job_name': args.job_name,
          'project': args.project_id,
          'no_save_main_session': True,
          'extra_packages': local_packages,
          'teardown_policy': 'TEARDOWN_ALWAYS',
      }
      opts = beam.pipeline.PipelineOptions(flags=[], **options)
      # Or use BlockingDataflowPipelineRunner
      p = beam.Pipeline('DataflowRunner', options=opts)
      make_prediction_pipeline(p, args)
      print(('Dataflow Job submitted, see Job %s at '
             'https://console.developers.google.com/dataflow?project=%s') %
            (options['job_name'], args.project_id))
      sys.stdout.flush()
      runner_results = p.run()
    finally:
      shutil.rmtree(tmpdir)
  else:
    p = beam.Pipeline('DirectRunner')
    make_prediction_pipeline(p, args)
    runner_results = p.run()

  return runner_results


if __name__ == '__main__':
  runner_results = main()
  runner_results.wait_until_finish()
