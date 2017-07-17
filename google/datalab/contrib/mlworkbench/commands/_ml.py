# Copyright 2017 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Platform library - ml cell magic."""
from __future__ import absolute_import
from __future__ import unicode_literals

try:
  import IPython
  import IPython.core.display
  import IPython.core.magic
except ImportError:
  raise Exception('This module can only be loaded in ipython.')

import argparse
import json
import os
import pandas as pd
import numpy as np
import shutil
import six
import subprocess
import tempfile
import textwrap
from tensorflow.python.lib.io import file_io
import urllib

import google.datalab
import google.datalab.bigquery as bq
from google.datalab import Context
import google.datalab.ml as datalab_ml
import google.datalab.utils.commands
import google.datalab.contrib.mlworkbench._local_predict as _local_predict
import google.datalab.contrib.mlworkbench._shell_process as _shell_process
import google.datalab.contrib.mlworkbench._archive as _archive


MLTOOLBOX_CODE_PATH = '/datalab/lib/pydatalab/solutionbox/code_free_ml/mltoolbox/code_free_ml/'


@IPython.core.magic.register_line_cell_magic
def ml(line, cell=None):
  """Implements the datalab cell magic for MLWorkbench operations.

  Args:
    line: the contents of the ml command line.
  Returns:
    The results of executing the cell.
  """
  parser = google.datalab.utils.commands.CommandParser(
      prog='%ml',
      description=textwrap.dedent("""\
          Execute MLWorkbench operations

          Use "%ml <command> -h" for help on a specific command.
      """))

  analyze_parser = parser.subcommand(
      'analyze',
      formatter_class=argparse.RawTextHelpFormatter,
      help='Analyze training data and generate stats, such as min/max/mean '
           'for numeric values, vocabulary for text columns.',
      epilog=textwrap.dedent("""\
          Example usage:

          %%ml analyze [--cloud]
          output: path/to/dir
          training_data:
            csv: path/to/csv
            schema:
              - name: serialId
                type: STRING
              - name: num1
                type: FLOAT
              - name: num2
                type: INTEGER
              - name: text1
                type: STRING
          features:
            serialId:
              transform: key
            num1:
              transform: scale
              value: 1
            num2:
              transform: identity
            text1:
              transform: bag_of_words

          Also supports in-notebook variables, such as:
          %%ml analyze --output path/to/dir
          training_data: $my_csv_dataset
          features: $features_def"""))

  analyze_parser.add_argument('--output', required=True,
                              help='path of output directory.')
  analyze_parser.add_argument('--cloud', action='store_true', default=False,
                              help='whether to run analysis in cloud or local.')
  analyze_parser.add_argument('--package', required=False,
                              help='A local or GCS tarball path to use as the source. '
                                   'If not set, the default source package will be used.')
  analyze_parser.add_cell_argument(
    'training_data',
    required=True,
    help=textwrap.dedent("""\
        training data. It is one of the following:
            csv (example "csv: file.csv"), or
            bigquery_table (example: "bigquery_table: project.dataset.table"), or
            bigquery_sql (example: "bigquery_sql: select * from table where num1 > 1.0"), or
            a variable defined as google.datalab.ml.CsvDataSet or
                google.datalab.ml.BigQueryDataSet."""))
  analyze_parser.add_cell_argument(
      'features',
      required=True,
      help=textwrap.dedent("""\
          features config indicating how to transform data into features. The
          list of supported transforms:
              "transform: identity"
                   does nothing (for numerical columns).
              "transform: scale
               value: x"
                   scale a numerical column to [-a, a]. If value is missing, x
                   defaults to 1.
              "transform: one_hot"
                   treats the string column as categorical and makes one-hot
                   encoding of it.
              "transform: embedding
               embedding_dim: d"
                   treats the string column as categorical and makes embeddings of
                   it with specified dimension size.
              "transform: bag_of_words"
                   treats the string column as text and make bag of words
                   transform of it.
              "transform: tfidf"
                   treats the string column as text and make TFIDF transform of it.
              "transform: image_to_vec"
                   from image gs url to embeddings.
              "transform: target"
                   denotes the column is the target. If the schema type of this
                   column is string, a one_hot encoding is automatically applied.
                   If numerical, an identity transform is automatically applied.
              "transform: key"
                   column contains metadata-like information and will be output
                   as-is in prediction."""))
  analyze_parser.set_defaults(func=_analyze)

  transform_parser = parser.subcommand(
      'transform',
      formatter_class=argparse.RawTextHelpFormatter,
      help='Transform the data into tf.example which is more efficient in training.',
      epilog=textwrap.dedent("""\
          Example usage:

          %%ml transform --cloud [--shuffle]
          analysis: path/to/analysis_output_folder
          output: path/to/dir
          prefix: my_filename
          batch_size: 100
          training_data:
            csv: path/to/csv
          cloud:
            num_workers: 3
            worker_machine_type: n1-standard-1
            project_id: my_project_id"""))
  transform_parser.add_argument('--analysis', required=True,
                                help='path of analysis output directory.')
  transform_parser.add_argument('--output', required=True,
                                help='path of output directory.')
  transform_parser.add_argument(
      '--prefix', required=True, metavar='NAME',
      help='The prefix of the output file name. The output files will be like '
           'NAME_00000_of_00005.tar.gz')
  transform_parser.add_argument('--cloud', action='store_true', default=False,
                                help='whether to run transform in cloud or local.')
  transform_parser.add_argument('--shuffle', action='store_true', default=False,
                                help='whether to shuffle the training data in output.')
  transform_parser.add_argument('--batch_size', type=int, default=100,
                                help='number of instances in a batch to process once. '
                                     'Larger batch is more efficient but may consume more memory.')
  transform_parser.add_argument('--package', required=False,
                                help='A local or GCS tarball path to use as the source. '
                                     'If not set, the default source package will be used.')
  transform_parser.add_cell_argument(
      'training_data',
      required=True,
      help=textwrap.dedent("""\
          Training data. A dict containing one of the following:
              csv (example: "csv: file.csv"), or
              bigquery_table (example: "bigquery_table: project.dataset.table"), or
              bigquery_sql (example: "bigquery_sql: select * from table where num1 > 1.0")"""))
  transform_parser.add_cell_argument(
      'cloud_config',
      help=textwrap.dedent("""\
          A dictionary of cloud config. All of them are optional.
              num_workers: Dataflow number of workers. If not set, DataFlow
                  service will determine the number.
              worker_machine_type: a machine name from
                  https://cloud.google.com/compute/docs/machine-types
                  If not given, the service uses the default machine type.
              project_id: id of the project to use for DataFlow service. If not set,
                  Datalab's default project (set by %%datalab project set) is used.
              job_name: Unique name for a Dataflow job to use. If not set, a
                  random name will be used."""))
  transform_parser.set_defaults(func=_transform)

  train_parser = parser.subcommand(
      'train',
      formatter_class=argparse.RawTextHelpFormatter,
      help='Train a model.',
      epilog=textwrap.dedent("""\
          Example usage:

          %%ml train --cloud
          analysis: path/to/analysis_output
          output: path/to/dir
          training_data:
            transformed: path/to/transformed/train
          evaluation_data:
            tranaformed: path/to/transformed/eval
          model_args:
            model: linear_regression
          cloud:
            region: us-central1"""))
  train_parser.add_argument('--analysis', required=True,
                            help='path of analysis output directory.')
  train_parser.add_argument('--output', required=True,
                            help='path of trained model directory.')
  train_parser.add_argument('--cloud', action='store_true', default=False,
                            help='whether to run training in cloud or local.')
  train_parser.add_argument('--package', required=False,
                            help='A local or GCS tarball path to use as the source. '
                                 'If not set, the default source package will be used.')
  train_parser.add_cell_argument(
      'training_data',
      required=True,
      help=textwrap.dedent("""\
          Training data. It is either raw csv file pattern, or transformed file pattern.
              For example:
              "training_data:
                csv: /path/to/csv/mycsv*.csv"

              or

              "training_data:
                transformed: /path/to/transformed-*" """))
  train_parser.add_cell_argument('evaluation_data', required=True,
                                 help='same as training_data.')

  package_model_help = subprocess.Popen(
      ['python', '-m', 'trainer.task', '--datalab-help'],
      cwd=MLTOOLBOX_CODE_PATH,
      stdout=subprocess.PIPE).communicate()[0]
  package_model_help = ('model_args: a dictionary of model specific args, including:\n\n' +
                        package_model_help.decode())
  train_parser.add_cell_argument('model_args', help=package_model_help)

  train_parser.add_cell_argument(
      'cloud_config',
      help=textwrap.dedent("""\
          A dictionary of cloud training config, including:
              job_id: the name of the job. If not provided, a default job name is created.
              region: see {url}
              runtime_version: see "region". Must be a string like '1.2'.
              scale_tier: see "region".""".format(
          url='https://cloud.google.com/sdk/gcloud/reference/ml-engine/jobs/submit/training')))
  train_parser.set_defaults(func=_train)

  predict_parser = parser.subcommand(
      'predict',
      formatter_class=argparse.RawTextHelpFormatter,
      help='Predict with local or deployed models. (Good for small datasets).',
      epilog=textwrap.dedent("""\
          Example usage:

          %%ml predict
          headers: key,num
          model: path/to/model
          prediction_data:
            - key1,value1
            - key2,value2

          Or, in another cell, define a list of dict:

          my_data = [{'key': 1, 'num': 1.2}, {'key': 2, 'num': 2.8}]

          Then:

          %%ml predict
          headers: key,num
          model: path/to/model
          prediction_data: $my_data"""))
  predict_parser.add_argument('--model', required=True,
                              help='The model path if not --cloud, or the id in '
                                   'the form of model.version if --cloud.')
  predict_parser.add_argument('--headers', required=True,
                              help='The comma seperated headers of the prediction data. '
                                   'Order must match the training order.')
  predict_parser.add_argument('--image_columns',
                              help='Comma seperated headers of image URL columns. '
                                   'Required if prediction data contains image URL columns.')
  predict_parser.add_argument('--no_show_image', action='store_true', default=False,
                              help='If not set, add a column of images in output.')
  predict_parser.add_argument('--cloud', action='store_true', default=False,
                              help='whether to run prediction in cloud or local.')
  predict_parser.add_cell_argument(
      'prediction_data',
      required=True,
      help=textwrap.dedent("""\
          Prediction data can be
              1) CSV lines in the input cell in yaml format or
              2) a local variable which is one of
                a) list of dict
                b) list of strings of csv lines
                c) a Pandas DataFrame"""))
  predict_parser.set_defaults(func=_predict)

  batch_predict_parser = parser.subcommand(
      'batch_predict',
      formatter_class=argparse.RawTextHelpFormatter,
      help='Batch prediction with local or deployed models. (Good for large datasets)',
      epilog=textwrap.dedent("""\

      Example usage:

      %%ml batch_predict [--cloud]
      model: path/to/model
      output: path/to/output
      format: csv
      prediction_data:
        csv: path/to/file_pattern"""))
  batch_predict_parser.add_argument('--model', required=True,
                                    help='The model path if not --cloud, or the id in '
                                         'the form of model.version if --cloud.')
  batch_predict_parser.add_argument('--output', required=True,
                                    help='The path of output directory with prediction results. '
                                         'If --cloud, it has to be GCS path.')
  batch_predict_parser.add_argument('--format',
                                    help='csv or json. For cloud run, '
                                         'the only supported format is json.')
  batch_predict_parser.add_argument('--batch_size', type=int, default=100,
                                    help='number of instances in a batch to process once. '
                                         'Larger batch is more efficient but may consume '
                                         'more memory. Only used in local run.')
  batch_predict_parser.add_argument('--cloud', action='store_true', default=False,
                                    help='whether to run prediction in cloud or local.')
  batch_predict_parser.add_cell_argument(
      'prediction_data',
      required=True,
      help='Data to predict with. Only csv is supported.')
  batch_predict_parser.add_cell_argument(
      'cloud_config',
      help=textwrap.dedent("""\
          A dictionary of cloud batch prediction config.
              job_id: the name of the job. If not provided, a default job name is created.
              region: see {url}
              max_worker_count: see reference in "region".""".format(
                  url='https://cloud.google.com/sdk/gcloud/reference/ml-engine/jobs/submit/prediction')))  # noqa
  batch_predict_parser.set_defaults(func=_batch_predict)

  return google.datalab.utils.commands.handle_magic_line(line, cell, parser)


def _abs_path(path):
  """Convert a non-GCS path to its absolute path.

  path can contain special filepath characters like '..', '*' and '.'.

  Example: If the current folder is /content/datalab/folder1 and path is
  '../folder2/files*', then this function returns the string
  '/content/datalab/folder2/files*'.

  This function is needed if using _shell_process.run_and_monitor() as that
  function runs a command in a different folder.

  Args:
    path: string.
  """
  if path.startswith('gs://'):
    return path
  return os.path.abspath(path)


def _create_json_file(tmpdir, data, filename):
  json_file = os.path.join(tmpdir, filename)
  with file_io.FileIO(json_file, 'w') as f:
    json.dump(data, f)
  return json_file


def _show_job_link(job):
  log_url_query_strings = {
    'project': Context.default().project_id,
    'resource': 'ml.googleapis.com/job_id/' + job.info['jobId']
  }
  log_url = 'https://console.developers.google.com/logs/viewer?' + \
      urllib.urlencode(log_url_query_strings)
  html = 'Job "%s" submitted.' % job.info['jobId']
  html += '<p>Click <a href="%s" target="_blank">here</a> to view cloud log. <br/>' % log_url
  IPython.display.display_html(html, raw=True)


def _analyze(args, cell):
  # For now, always run python2. If needed we can run python3 when the current kernel
  # is py3. Since now our transform cannot work on py3 anyway, I would rather run
  # everything with python2.
  cmd_args = ['python', 'analyze.py', '--output', _abs_path(args['output'])]
  if args['cloud']:
    cmd_args.append('--cloud')

  training_data = args['training_data']
  if args['cloud']:
    tmpdir = os.path.join(args['output'], 'tmp')
  else:
    tmpdir = tempfile.mkdtemp()

  try:
    if isinstance(training_data, dict):
      if 'csv' in training_data and 'schema' in training_data:
        schema = training_data['schema']
        schema_file = _create_json_file(tmpdir, schema, 'schema.json')
        cmd_args.append('--csv=' + _abs_path(training_data['csv']))
        cmd_args.extend(['--schema', schema_file])
      elif 'bigquery_table' in training_data:
        cmd_args.extend(['--bigquery', training_data['bigquery_table']])
      elif 'bigquery_sql' in training_data:
        # see https://cloud.google.com/bigquery/querying-data#temporary_and_permanent_tables
        print('Creating temporary table that will be deleted in 24 hours')
        r = bq.Query(training_data['bigquery_sql']).execute().result()
        cmd_args.extend(['--bigquery', r.full_name])
      else:
        raise ValueError('Invalid training_data dict. '
                         'Requires either "csv_file_pattern" and "csv_schema", or "bigquery".')
    elif isinstance(training_data, google.datalab.ml.CsvDataSet):
      schema_file = _create_json_file(tmpdir, training_data.schema, 'schema.json')
      for file_name in training_data.input_files:
        cmd_args.append('--csv=' + _abs_path(file_name))

      cmd_args.extend(['--schema', schema_file])
    elif isinstance(training_data, google.datalab.ml.BigQueryDataSet):
      # TODO: Support query too once command line supports query.
      cmd_args.extend(['--bigquery', training_data.table])
    else:
      raise ValueError('Invalid training data. Requires either a dict, '
                       'a google.datalab.ml.CsvDataSet, or a google.datalab.ml.BigQueryDataSet.')

    features = args['features']
    features_file = _create_json_file(tmpdir, features, 'features.json')
    cmd_args.extend(['--features', features_file])

    if args['package']:
      code_path = os.path.join(tmpdir, 'package')
      _archive.extract_archive(args['package'], code_path)
    else:
      code_path = MLTOOLBOX_CODE_PATH

    _shell_process.run_and_monitor(cmd_args, os.getpid(), cwd=code_path)
  finally:
    file_io.delete_recursively(tmpdir)


def _transform(args, cell):
  if args['cloud_config'] and not args['cloud']:
    raise ValueError('"cloud_config" is provided but no "--cloud". '
                     'Do you want local run or cloud run?')

  cmd_args = ['python', 'transform.py',
              '--output', _abs_path(args['output']),
              '--analysis', _abs_path(args['analysis']),
              '--prefix', args['prefix']]
  if args['cloud']:
    cmd_args.append('--cloud')
    cmd_args.append('--async')
  if args['shuffle']:
    cmd_args.append('--shuffle')
  if args['batch_size']:
    cmd_args.extend(['--batch-size', str(args['batch_size'])])

  training_data = args['training_data']
  if isinstance(training_data, dict):
    if 'csv' in training_data:
      cmd_args.append('--csv=' + _abs_path(training_data['csv']))
    elif 'bigquery_table' in training_data:
      cmd_args.extend(['--bigquery', training_data['bigquery_table']])
    elif 'bigquery_sql' in training_data:
        # see https://cloud.google.com/bigquery/querying-data#temporary_and_permanent_tables
        print('Creating temporary table that will be deleted in 24 hours')
        r = bq.Query(training_data['bigquery_sql']).execute().result()
        cmd_args.extend(['--bigquery', r.full_name])
    else:
      raise ValueError('Invalid training_data dict. '
                       'Requires either "csv", or "bigquery_talbe", or '
                       '"bigquery_sql".')
  elif isinstance(training_data, google.datalab.ml.CsvDataSet):
    for file_name in training_data.input_files:
      cmd_args.append('--csv=' + _abs_path(file_name))
  elif isinstance(training_data, google.datalab.ml.BigQueryDataSet):
    cmd_args.extend(['--bigquery', training_data.table])
  else:
    raise ValueError('Invalid training data. Requires either a dict, '
                     'a google.datalab.ml.CsvDataSet, or a google.datalab.ml.BigQueryDataSet.')

  cloud_config = args['cloud_config']
  if cloud_config:
    google.datalab.utils.commands.validate_config(
        cloud_config,
        required_keys=[],
        optional_keys=['num_workers', 'worker_machine_type', 'project_id', 'job_name'])
    if 'num_workers' in cloud_config:
      cmd_args.extend(['--num-workers', str(cloud_config['num_workers'])])
    if 'worker_machine_type' in cloud_config:
      cmd_args.extend(['--worker-machine-type', cloud_config['worker_machine_type']])
    if 'project_id' in cloud_config:
      cmd_args.extend(['--project-id', cloud_config['project_id']])
    else:
      cmd_args.extend(['--project-id', google.datalab.Context.default().project_id])
    if 'job_name' in cloud_config:
      cmd_args.extend(['--job-name', cloud_config['job_name']])

  try:
    tmpdir = None
    if args['package']:
      tmpdir = tempfile.mkdtemp()
      code_path = os.path.join(tmpdir, 'package')
      _archive.extract_archive(args['package'], code_path)
    else:
      code_path = MLTOOLBOX_CODE_PATH
    _shell_process.run_and_monitor(cmd_args, os.getpid(), cwd=code_path)
  finally:
    if tmpdir:
      shutil.rmtree(tmpdir)


def _train(args, cell):
  if args['cloud_config'] and not args['cloud']:
    raise ValueError('"cloud_config" is provided but no "--cloud". '
                     'Do you want local run or cloud run?')

  job_args = ['--job-dir', _abs_path(args['output']),
              '--analysis', _abs_path(args['analysis'])]

  def _process_train_eval_data(data, arg_name, job_args):
    if isinstance(data, dict):
      if 'csv' in data:
        job_args.append(arg_name + '=' + _abs_path(data['csv']))
        if '--transform' not in job_args:
          job_args.append('--transform')
      elif 'transformed' in data:
        job_args.append(arg_name + '=' + _abs_path(data['transformed']))
      else:
        raise ValueError('Invalid training_data dict. '
                         'Requires either "csv" or "transformed".')
    elif isinstance(data, google.datalab.ml.CsvDataSet):
      for file_name in data.input_files:
        job_args.append(arg_name + '=' + _abs_path(file_name))
    else:
      raise ValueError('Invalid training data. Requires either a dict, or '
                       'a google.datalab.ml.CsvDataSet')

  _process_train_eval_data(args['training_data'], '--train', job_args)
  _process_train_eval_data(args['evaluation_data'], '--eval', job_args)

  # TODO(brandondutra) document that any model_args that are file paths must
  # be given as an absolute path
  if args['model_args']:
    for k, v in six.iteritems(args['model_args']):
      job_args.extend(['--' + k, str(v)])

  try:
    tmpdir = None
    if args['package']:
      tmpdir = tempfile.mkdtemp()
      code_path = os.path.join(tmpdir, 'package')
      _archive.extract_archive(args['package'], code_path)
    else:
      code_path = MLTOOLBOX_CODE_PATH

    if args['cloud']:
      cloud_config = args['cloud_config']
      if not args['output'].startswith('gs://'):
        raise ValueError('Cloud training requires a GCS (starting with "gs://") output.')

      staging_tarball = os.path.join(args['output'], 'staging', 'trainer.tar.gz')
      datalab_ml.package_and_copy(code_path,
                                  os.path.join(code_path, 'setup.py'),
                                  staging_tarball)
      job_request = {
          'package_uris': [staging_tarball],
          'python_module': 'trainer.task',
          'job_dir': args['output'],
          'args': job_args,
      }
      job_request.update(cloud_config)
      job_id = cloud_config.get('job_id', None)
      job = datalab_ml.Job.submit_training(job_request, job_id)
      _show_job_link(job)
    else:
      cmd_args = ['python', '-m', 'trainer.task'] + job_args
      _shell_process.run_and_monitor(cmd_args, os.getpid(), cwd=code_path)
  finally:
    if tmpdir:
      shutil.rmtree(tmpdir)


def _predict(args, cell):
  headers = args['headers'].split(',')
  img_cols = args['image_columns'].split(',') if args['image_columns'] else []

  data = args['prediction_data']
  df = _local_predict.get_prediction_results(
      args['model'], data, headers, img_cols=img_cols, cloud=args['cloud'],
      show_image=not args['no_show_image'])

  def _show_img(img_bytes):
    return '<img src="data:image/png;base64,' + img_bytes + '" />'

  def _truncate_text(text):
    return (text[:37] + '...') if isinstance(text, six.string_types) and len(text) > 40 else text

  # Truncate text explicitly here because we will set display.max_colwidth to -1.
  # This applies to images to but images will be overriden with "_show_img()" later.
  formatters = {x: _truncate_text for x in df.columns if df[x].dtype == np.object}
  if not args['no_show_image'] and img_cols:
    formatters.update({x + '_image': _show_img for x in img_cols})

  # Set display.max_colwidth to -1 so we can display images.
  old_width = pd.get_option('display.max_colwidth')
  pd.set_option('display.max_colwidth', -1)
  try:
    IPython.display.display(IPython.display.HTML(
        df.to_html(formatters=formatters, escape=False, index=False)))
  finally:
    pd.set_option('display.max_colwidth', old_width)


def _batch_predict(args, cell):
  if args['cloud_config'] and not args['cloud']:
    raise ValueError('"cloud_config" is provided but no "--cloud". '
                     'Do you want local run or cloud run?')

  if args['cloud']:
    parts = args['model'].split('.')
    if len(parts) != 2:
      raise ValueError('Invalid model name for cloud prediction. Use "model.version".')

    version_name = ('projects/%s/models/%s/versions/%s' %
                    (Context.default().project_id, parts[0], parts[1]))

    cloud_config = args['cloud_config'] or {}
    job_id = cloud_config.pop('job_id', None)
    job_request = {
      'version_name': version_name,
      'data_format': 'TEXT',
      'input_paths': file_io.get_matching_files(args['prediction_data']['csv']),
      'output_path': args['output'],
    }
    job_request.update(cloud_config)
    job = datalab_ml.Job.submit_batch_prediction(job_request, job_id)
    _show_job_link(job)
  else:
    print('local prediction...')
    _local_predict.local_batch_predict(args['model'],
                                       args['prediction_data']['csv'],
                                       args['output'],
                                       args['format'],
                                       args['batch_size'])
    print('done.')
