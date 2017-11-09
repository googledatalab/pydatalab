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
import collections
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil
import six
from skimage.segmentation import mark_boundaries
import subprocess
import tempfile
import textwrap
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import urllib

import google.datalab
from google.datalab import Context
import google.datalab.ml as datalab_ml
import google.datalab.utils.commands
import google.datalab.contrib.mlworkbench._local_predict as _local_predict
import google.datalab.contrib.mlworkbench._shell_process as _shell_process
import google.datalab.contrib.mlworkbench._archive as _archive
import google.datalab.contrib.mlworkbench._prediction_explainer as _prediction_explainer


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

  dataset_parser = parser.subcommand(
      'dataset',
      formatter_class=argparse.RawTextHelpFormatter,
      help='Create or explore datasets.')
  dataset_sub_commands = dataset_parser.add_subparsers(dest='command')
  dataset_create_parser = dataset_sub_commands.add_parser(
      'create', help='Create datasets', formatter_class=argparse.RawTextHelpFormatter,
      epilog=textwrap.dedent("""\
          Example usage:

          %%ml dataset
          name: mydata
          format: csv
          train: path/to/train.csv
          eval: path/to/eval.csv
          schema:
            - name: news_label
              type: STRING
            - name: text
              type: STRING"""))

  dataset_create_parser.add_argument('--name', required=True,
                                     help='the name of the dataset to define. ')
  dataset_create_parser.add_argument('--format', required=True,
                                     choices=['csv', 'bigquery', 'transformed'],
                                     help='The format of the data.')
  dataset_create_parser.add_argument('--train', required=True,
                                     help='The path of the training file pattern if format ' +
                                          'is csv or transformed, or table name if format ' +
                                          'is bigquery.')
  dataset_create_parser.add_argument('--eval', required=True,
                                     help='The path of the eval file pattern if format ' +
                                          'is csv or transformed, or table name if format ' +
                                          'is bigquery.')
  dataset_create_parser.add_cell_argument('schema',
                                          help='yaml representation of CSV schema, or path to ' +
                                          'schema file. Only needed if format is csv.')
  dataset_create_parser.set_defaults(func=_dataset_create)

  dataset_explore_parser = dataset_sub_commands.add_parser(
      'explore', help='Explore training data.')
  dataset_explore_parser.add_argument('--name', required=True,
                                      help='The name of the dataset to explore.')

  dataset_explore_parser.add_argument('--overview', action='store_true', default=False,
                                      help='Plot overview of sampled data. Set "sample_size" ' +
                                           'to change the default sample size.')
  dataset_explore_parser.add_argument('--facets', action='store_true', default=False,
                                      help='Plot facets view of sampled data. Set ' +
                                           '"sample_size" to change the default sample size.')
  dataset_explore_parser.add_argument('--sample_size', type=int, default=1000,
                                      help='sample size for overview or facets view. Only ' +
                                           'used if either --overview or --facets is set.')
  dataset_explore_parser.set_defaults(func=_dataset_explore)

  analyze_parser = parser.subcommand(
      'analyze',
      formatter_class=argparse.RawTextHelpFormatter,
      help='Analyze training data and generate stats, such as min/max/mean '
           'for numeric values, vocabulary for text columns.',
      epilog=textwrap.dedent("""\
          Example usage:

          %%ml analyze [--cloud]
          output: path/to/dir
          data: $mydataset
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
    'data',
    required=True,
    help="""Training data. A dataset defined by "%%ml dataset".""")
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
              "transform: image_to_vec
               checkpoint: gs://b/o"
                   from image gs url to embeddings. "checkpoint" is a inception v3
                   checkpoint. If absent, a default checkpoint is used.
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

          %%ml transform [--cloud] [--shuffle]
          analysis: path/to/analysis_output_folder
          output: path/to/dir
          batch_size: 100
          data: $mydataset
          cloud:
            num_workers: 3
            worker_machine_type: n1-standard-1
            project_id: my_project_id"""))
  transform_parser.add_argument('--analysis', required=True,
                                help='path of analysis output directory.')
  transform_parser.add_argument('--output', required=True,
                                help='path of output directory.')
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
      'data',
      required=True,
      help="""Training data. A dataset defined by "%%ml dataset".""")
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

          %%ml train [--cloud]
          analysis: path/to/analysis_output
          output: path/to/dir
          data: $mydataset
          model_args:
            model: linear_regression
          cloud_config:
            region: us-central1"""))
  train_parser.add_argument('--analysis', required=True,
                            help='path of analysis output directory.')
  train_parser.add_argument('--output', required=True,
                            help='path of trained model directory.')
  train_parser.add_argument('--cloud', action='store_true', default=False,
                            help='whether to run training in cloud or local.')
  train_parser.add_argument('--notb', action='store_true', default=False,
                            help='If set, tensorboard is not automatically started.')
  train_parser.add_argument('--package', required=False,
                            help='A local or GCS tarball path to use as the source. '
                                 'If not set, the default source package will be used.')
  train_parser.add_cell_argument(
      'data',
      required=True,
      help="""Training data. A dataset defined by "%%ml dataset".""")

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
          data:
            - key1,value1
            - key2,value2

          Or, in another cell, define a list of dict:

          my_data = [{'key': 1, 'num': 1.2}, {'key': 2, 'num': 2.8}]

          Then:

          %%ml predict
          headers: key,num
          model: path/to/model
          data: $my_data"""))
  predict_parser.add_argument('--model', required=True,
                              help='The model path.')
  predict_parser.add_argument('--no_show_image', action='store_true', default=False,
                              help='If not set, add a column of images in output.')
  predict_parser.add_cell_argument(
      'data',
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
      data:
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
      'data',
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

  explain_parser = parser.subcommand(
      'explain',
      formatter_class=argparse.RawTextHelpFormatter,
      help='Explain a prediction with LIME tool.')
  explain_parser.add_argument('--type', default='all', choices=['text', 'image', 'tabular', 'all'],
                              help='the type of column to explain.')
  explain_parser.add_argument('--algorithm', choices=['lime', 'ig'], default='lime',
                              help='"lime" is the open sourced project for prediction explainer.' +
                                   '"ig" means integrated gradients and currently only applies ' +
                                   'to image.')
  explain_parser.add_argument('--model', required=True,
                              help='path of the model directory used for prediction.')
  explain_parser.add_argument('--labels', required=True,
                              help='comma separated labels to explain.')
  explain_parser.add_argument('--column_name',
                              help='the name of the column to explain. Optional if text type ' +
                                   'and there is only one text column, or image type and ' +
                                   'there is only one image column.')
  explain_parser.add_cell_argument('data', required=True,
                                   help='Prediction Data. Can be a csv line, or a dict.')
  explain_parser.add_cell_argument('training_data',
                                   help='A csv or bigquery dataset defined by %%ml dataset. ' +
                                        'Used by tabular explainer only to determine the ' +
                                        'distribution of numeric and categorical values. ' +
                                        'Suggest using original training dataset.')

  # options specific for lime
  explain_parser.add_argument('--num_features', type=int,
                              help='number of features to analyze. In text, it is number of ' +
                                   'words. In image, it is number of areas. For lime only.')
  explain_parser.add_argument('--num_samples', type=int,
                              help='size of the neighborhood to learn the linear model. ' +
                                   'For lime only.')
  explain_parser.add_argument('--hide_color', type=int, default=0,
                              help='the color to use for perturbed area. If -1, average of ' +
                                   'each channel is used for each channel. For image only.')
  explain_parser.add_argument('--include_negative', action='store_true', default=False,
                              help='whether to show only positive areas. For lime image only.')
  explain_parser.add_argument('--overview', action='store_true', default=False,
                              help='whether to show overview instead of details view.' +
                                   'For lime text and tabular only.')
  explain_parser.add_argument('--batch_size', type=int, default=100,
                              help='size of batches passed to prediction. For lime only.')

  # options specific for integrated gradients
  explain_parser.add_argument('--num_gradients', type=int, default=50,
                              help='the number of scaled images to get gradients from. Larger ' +
                                   'number usually produces better results but slower.')
  explain_parser.add_argument('--percent_show', type=int, default=10,
                              help='the percentage of top impactful pixels to show.')

  explain_parser.set_defaults(func=_explain)

  tensorboard_parser = parser.subcommand(
      'tensorboard',
      formatter_class=argparse.RawTextHelpFormatter,
      help='Start/stop/list TensorBoard instances.')
  tensorboard_sub_commands = tensorboard_parser.add_subparsers(dest='command')

  tensorboard_start_parser = tensorboard_sub_commands.add_parser(
      'start', help='Start a tensorboard instance.')
  tensorboard_start_parser.add_argument('--logdir', required=True,
                                        help='The local or GCS logdir path.')
  tensorboard_start_parser.set_defaults(func=_tensorboard_start)

  tensorboard_stop_parser = tensorboard_sub_commands.add_parser(
      'stop', help='Stop a tensorboard instance.')
  tensorboard_stop_parser.add_argument('--pid', required=True, type=int,
                                       help='The pid of the tensorboard instance.')
  tensorboard_stop_parser.set_defaults(func=_tensorboard_stop)

  tensorboard_list_parser = tensorboard_sub_commands.add_parser(
      'list', help='List tensorboard instances.')
  tensorboard_list_parser.set_defaults(func=_tensorboard_list)

  evaluate_parser = parser.subcommand(
      'evaluate',
      formatter_class=argparse.RawTextHelpFormatter,
      help='Analyze model evaluation results, such as confusion matrix, ROC, RMSE.')
  evaluate_sub_commands = evaluate_parser.add_subparsers(dest='command')

  def _add_data_params_for_evaluate(parser):
    parser.add_argument('--csv', help='csv file path patterns.')
    parser.add_argument('--headers',
                        help='csv file headers. Required if csv is specified and ' +
                             'predict_results_schema.json does not exist in the same directory.')
    parser.add_argument('--bigquery',
                        help='can be bigquery table, query as a string, or ' +
                             'a pre-defined query (%%bq query --name).')

  evaluate_cm_parser = evaluate_sub_commands.add_parser(
      'confusion_matrix', help='Get confusion matrix from evaluation results.')
  _add_data_params_for_evaluate(evaluate_cm_parser)
  evaluate_cm_parser.add_argument('--plot', action='store_true', default=False,
                                  help='Whether to plot confusion matrix as graph.')
  evaluate_cm_parser.add_argument('--size', type=int, default=10,
                                  help='The size of the confusion matrix.')
  evaluate_cm_parser.set_defaults(func=_evaluate_cm)

  evaluate_accuracy_parser = evaluate_sub_commands.add_parser(
      'accuracy', help='Get accuracy results from classification evaluation results.')
  _add_data_params_for_evaluate(evaluate_accuracy_parser)
  evaluate_accuracy_parser.set_defaults(func=_evaluate_accuracy)

  evaluate_pr_parser = evaluate_sub_commands.add_parser(
      'precision_recall', help='Get precision recall metrics from evaluation results.')
  _add_data_params_for_evaluate(evaluate_pr_parser)
  evaluate_pr_parser.add_argument('--plot', action='store_true', default=False,
                                  help='Whether to plot precision recall as graph.')
  evaluate_pr_parser.add_argument('--num_thresholds', type=int, default=20,
                                  help='Number of thresholds which determines how many ' +
                                       'points in the graph.')
  evaluate_pr_parser.add_argument('--target_class', required=True,
                                  help='The target class to determine correctness of ' +
                                       'a prediction.')
  evaluate_pr_parser.add_argument('--probability_column',
                                  help='The name of the column holding the probability ' +
                                       'value of the target class. If absent, the value ' +
                                       'of target class is used.')
  evaluate_pr_parser.set_defaults(func=_evaluate_pr)

  evaluate_roc_parser = evaluate_sub_commands.add_parser(
      'roc', help='Get ROC metrics from evaluation results.')
  _add_data_params_for_evaluate(evaluate_roc_parser)
  evaluate_roc_parser.add_argument('--plot', action='store_true', default=False,
                                   help='Whether to plot ROC as graph.')
  evaluate_roc_parser.add_argument('--num_thresholds', type=int, default=20,
                                   help='Number of thresholds which determines how many ' +
                                        'points in the graph.')
  evaluate_roc_parser.add_argument('--target_class', required=True,
                                   help='The target class to determine correctness of ' +
                                        'a prediction.')
  evaluate_roc_parser.add_argument('--probability_column',
                                   help='The name of the column holding the probability ' +
                                        'value of the target class. If absent, the value ' +
                                        'of target class is used.')
  evaluate_roc_parser.set_defaults(func=_evaluate_roc)

  evaluate_regression_parser = evaluate_sub_commands.add_parser(
      'regression', help='Get regression metrics from evaluation results.')
  _add_data_params_for_evaluate(evaluate_regression_parser)
  evaluate_regression_parser.set_defaults(func=_evaluate_regression)

  model_parser = parser.subcommand(
      'model',
      help='Models and versions management such as deployment, deletion, listing.')
  model_sub_commands = model_parser.add_subparsers(dest='command')
  model_list_parser = model_sub_commands.add_parser(
      'list', help='List models and versions.')
  model_list_parser.add_argument('--name',
                                 help='If absent, list all models of specified or current ' +
                                      'project. If provided, list all versions of the ' +
                                      'model.')
  model_list_parser.add_argument('--project',
                                 help='The project to list model(s) or version(s). If absent, ' +
                                      'use Datalab\'s default project.')
  model_list_parser.set_defaults(func=_model_list)

  model_delete_parser = model_sub_commands.add_parser(
      'delete', help='Delete models or versions.')
  model_delete_parser.add_argument('--name', required=True,
                                   help='If no "." in the name, try deleting the specified ' +
                                        'model. If "model.version" is provided, try deleting ' +
                                        'the specified version.')
  model_delete_parser.add_argument('--project',
                                   help='The project to delete model or version. If absent, ' +
                                        'use Datalab\'s default project.')
  model_delete_parser.set_defaults(func=_model_delete)

  model_deploy_parser = model_sub_commands.add_parser(
      'deploy', help='Deploy a model version.')
  model_deploy_parser.add_argument('--name', required=True,
                                   help='Must be model.version to indicate the model ' +
                                        'and version name to deploy.')
  model_deploy_parser.add_argument('--path', required=True,
                                   help='The GCS path of the model to be deployed.')
  model_deploy_parser.add_argument('--runtime_version',
                                   help='The TensorFlow version to use for this model. ' +
                                        'For example, "1.2.1". If absent, the current ' +
                                        'TensorFlow version installed in Datalab will be used.')
  model_deploy_parser.add_argument('--project',
                                   help='The project to deploy a model version. If absent, ' +
                                        'use Datalab\'s default project.')
  model_deploy_parser.set_defaults(func=_model_deploy)

  return google.datalab.utils.commands.handle_magic_line(line, cell, parser)


DataSet = collections.namedtuple('DataSet', ['train', 'eval'])


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


def get_dataset_from_arg(dataset_arg):
  if isinstance(dataset_arg, DataSet):
    return dataset_arg

  if isinstance(dataset_arg, six.string_types):
    return google.datalab.utils.commands.notebook_environment()[dataset_arg]

  raise ValueError('Invalid dataset reference "%s". ' % dataset_arg +
                   'Expect a dataset defined with "%%ml dataset create".')


def _analyze(args, cell):
  # For now, always run python2. If needed we can run python3 when the current kernel
  # is py3. Since now our transform cannot work on py3 anyway, I would rather run
  # everything with python2.
  cmd_args = ['python', 'analyze.py', '--output', _abs_path(args['output'])]
  if args['cloud']:
    cmd_args.append('--cloud')

  training_data = get_dataset_from_arg(args['data'])

  if args['cloud']:
    tmpdir = os.path.join(args['output'], 'tmp')
  else:
    tmpdir = tempfile.mkdtemp()

  try:
    if isinstance(training_data.train, datalab_ml.CsvDataSet):
      csv_data = training_data.train
      schema_file = _create_json_file(tmpdir, csv_data.schema, 'schema.json')
      for file_name in csv_data.input_files:
        cmd_args.append('--csv=' + _abs_path(file_name))
      cmd_args.extend(['--schema', schema_file])
    elif isinstance(training_data.train, datalab_ml.BigQueryDataSet):
      bq_data = training_data.train
      cmd_args.extend(['--bigquery', bq_data.table])
    else:
      raise ValueError('Unexpected training data type. Only csv or bigquery are supported.')

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
              '--analysis', _abs_path(args['analysis'])]
  if args['cloud']:
    cmd_args.append('--cloud')
    cmd_args.append('--async')
  if args['shuffle']:
    cmd_args.append('--shuffle')
  if args['batch_size']:
    cmd_args.extend(['--batch-size', str(args['batch_size'])])

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
    if 'job_name' in cloud_config:
      cmd_args.extend(['--job-name', cloud_config['job_name']])

  if args['cloud'] and (not cloud_config or 'project_id' not in cloud_config):
    cmd_args.extend(['--project-id', google.datalab.Context.default().project_id])

  training_data = get_dataset_from_arg(args['data'])
  data_names = ('train', 'eval')
  for name in data_names:
    cmd_args_copy = list(cmd_args)
    if isinstance(getattr(training_data, name), datalab_ml.CsvDataSet):
      for file_name in getattr(training_data, name).input_files:
        cmd_args_copy.append('--csv=' + _abs_path(file_name))
    elif isinstance(getattr(training_data, name), datalab_ml.BigQueryDataSet):
      cmd_args_copy.extend(['--bigquery', getattr(training_data, name).table])
    else:
      raise ValueError('Unexpected training data type. Only csv or bigquery are supported.')

    cmd_args_copy.extend(['--prefix', name])
    try:
      tmpdir = None
      if args['package']:
        tmpdir = tempfile.mkdtemp()
        code_path = os.path.join(tmpdir, 'package')
        _archive.extract_archive(args['package'], code_path)
      else:
        code_path = MLTOOLBOX_CODE_PATH
      _shell_process.run_and_monitor(cmd_args_copy, os.getpid(), cwd=code_path)
    finally:
      if tmpdir:
        shutil.rmtree(tmpdir)


def _train(args, cell):
  if args['cloud_config'] and not args['cloud']:
    raise ValueError('"cloud_config" is provided but no "--cloud". '
                     'Do you want local run or cloud run?')

  job_args = ['--job-dir', _abs_path(args['output']),
              '--analysis', _abs_path(args['analysis'])]

  training_data = get_dataset_from_arg(args['data'])
  data_names = ('train', 'eval')
  for name in data_names:
    if (isinstance(getattr(training_data, name), datalab_ml.CsvDataSet) or
       isinstance(getattr(training_data, name), datalab_ml.TransformedDataSet)):
      for file_name in getattr(training_data, name).input_files:
        job_args.append('--%s=%s' % (name, _abs_path(file_name)))
    else:
      raise ValueError('Unexpected training data type. ' +
                       'Only csv and transformed type are supported.')

  if isinstance(training_data.train, datalab_ml.CsvDataSet):
    job_args.append('--transform')

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
      if not args['notb']:
        datalab_ml.TensorBoard.start(args['output'])
    else:
      cmd_args = ['python', '-m', 'trainer.task'] + job_args
      if not args['notb']:
        datalab_ml.TensorBoard.start(args['output'])
      _shell_process.run_and_monitor(cmd_args, os.getpid(), cwd=code_path)
  finally:
    if tmpdir:
      shutil.rmtree(tmpdir)


def _predict(args, cell):
  schema, features = _local_predict.get_model_schema_and_features(args['model'])
  headers = [x['name'] for x in schema]
  img_cols = []
  for k, v in six.iteritems(features):
    if v['transform'] in ['image_to_vec']:
      img_cols.append(v['source_column'])

  data = args['data']
  df = _local_predict.get_prediction_results(
      args['model'], data, headers, img_cols=img_cols, cloud=False,
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
    job_request = {
      'data_format': 'TEXT',
      'input_paths': file_io.get_matching_files(args['data']['csv']),
      'output_path': args['output'],
    }
    if args['model'].startswith('gs://'):
      job_request['uri'] = args['model']
    else:
      parts = args['model'].split('.')
      if len(parts) != 2:
        raise ValueError('Invalid model name for cloud prediction. Use "model.version".')

      version_name = ('projects/%s/models/%s/versions/%s' %
                      (Context.default().project_id, parts[0], parts[1]))
      job_request['version_name'] = version_name

    cloud_config = args['cloud_config'] or {}
    job_id = cloud_config.pop('job_id', None)
    job_request.update(cloud_config)
    job = datalab_ml.Job.submit_batch_prediction(job_request, job_id)
    _show_job_link(job)
  else:
    print('local prediction...')
    _local_predict.local_batch_predict(args['model'],
                                       args['data']['csv'],
                                       args['output'],
                                       args['format'],
                                       args['batch_size'])
    print('done.')


# Helper classes for explainer. Each for is for a combination
# of algorithm (LIME, IG) and type (text, image, tabular)
# ===========================================================
class _TextLimeExplainerInstance(object):

  def __init__(self, explainer, labels, args):
    num_features = args['num_features'] if args['num_features'] else 10
    num_samples = args['num_samples'] if args['num_samples'] else 5000
    self._exp = explainer.explain_text(
        labels, args['data'], column_name=args['column_name'],
        num_features=num_features, num_samples=num_samples)
    self._col_name = args['column_name'] if args['column_name'] else explainer._text_columns[0]
    self._show_overview = args['overview']

  def visualize(self, label_index):
    if self._show_overview:
      IPython.display.display(
          IPython.display.HTML('<br/>  Text Column "<b>%s</b>"<br/>' % self._col_name))
      self._exp.show_in_notebook(labels=[label_index])
    else:
      fig = self._exp.as_pyplot_figure(label=label_index)
      # Clear original title set by lime.
      plt.title('')
      fig.suptitle('Text Column "%s"' % self._col_name, fontsize=16)
      plt.close(fig)
      IPython.display.display(fig)


class _ImageLimeExplainerInstance(object):

  def __init__(self, explainer, labels, args):
    num_samples = args['num_samples'] if args['num_samples'] else 300
    hide_color = None if args['hide_color'] == -1 else args['hide_color']
    self._exp = explainer.explain_image(
        labels, args['data'], column_name=args['column_name'],
        num_samples=num_samples, batch_size=args['batch_size'], hide_color=hide_color)
    self._labels = labels
    self._positive_only = not args['include_negative']
    self._num_features = args['num_features'] if args['num_features'] else 3
    self._col_name = args['column_name'] if args['column_name'] else explainer._image_columns[0]

  def visualize(self, label_index):
    image, mask = self._exp.get_image_and_mask(
        label_index,
        positive_only=self._positive_only,
        num_features=self._num_features, hide_rest=False)
    fig = plt.figure()
    fig.suptitle('Image Column "%s"' % self._col_name, fontsize=16)
    plt.grid(False)
    plt.imshow(mark_boundaries(image, mask))
    plt.close(fig)
    IPython.display.display(fig)


class _ImageIgExplainerInstance(object):

  def __init__(self, explainer, labels, args):
    self._raw_image, self._analysis_images = explainer.probe_image(
        labels, args['data'], column_name=args['column_name'],
        num_scaled_images=args['num_gradients'], top_percent=args['percent_show'])
    self._labels = labels
    self._col_name = args['column_name'] if args['column_name'] else explainer._image_columns[0]

  def visualize(self, label_index):
    # Show both resized raw image and analyzed image.
    fig = plt.figure()
    fig.suptitle('Image Column "%s"' % self._col_name, fontsize=16)
    plt.grid(False)
    plt.imshow(self._analysis_images[label_index])
    plt.close(fig)
    IPython.display.display(fig)


class _TabularLimeExplainerInstance(object):

  def __init__(self, explainer, labels, args):
    if not args['training_data']:
      raise ValueError('tabular explanation requires training_data to determine ' +
                       'values distribution.')

    training_data = get_dataset_from_arg(args['training_data'])
    if (not isinstance(training_data.train, datalab_ml.CsvDataSet) and
       not isinstance(training_data.train, datalab_ml.BigQueryDataSet)):
      raise ValueError('Require csv or bigquery dataset.')

    sample_size = min(training_data.train.size, 10000)
    training_df = training_data.train.sample(sample_size)
    num_features = args['num_features'] if args['num_features'] else 5
    self._exp = explainer.explain_tabular(training_df, labels, args['data'],
                                          num_features=num_features)
    self._show_overview = args['overview']

  def visualize(self, label_index):
    if self._show_overview:
      IPython.display.display(
          IPython.display.HTML('<br/>All Categorical and Numeric Columns<br/>'))
      self._exp.show_in_notebook(labels=[label_index])
    else:
      fig = self._exp.as_pyplot_figure(label=label_index)
      # Clear original title set by lime.
      plt.title('')
      fig.suptitle('  All Categorical and Numeric Columns', fontsize=16)
      plt.close(fig)
      IPython.display.display(fig)

# End of Explainer Helper Classes
# ===================================================


def _explain(args, cell):

  explainer = _prediction_explainer.PredictionExplainer(args['model'])
  labels = args['labels'].split(',')
  instances = []
  if args['type'] == 'all':
    if explainer._numeric_columns or explainer._categorical_columns:
      instances.append(_TabularLimeExplainerInstance(explainer, labels, args))
    for col_name in explainer._text_columns:
      args['column_name'] = col_name
      instances.append(_TextLimeExplainerInstance(explainer, labels, args))
    for col_name in explainer._image_columns:
      args['column_name'] = col_name
      if args['algorithm'] == 'lime':
        instances.append(_ImageLimeExplainerInstance(explainer, labels, args))
      elif args['algorithm'] == 'ig':
        instances.append(_ImageIgExplainerInstance(explainer, labels, args))

  elif args['type'] == 'text':
    instances.append(_TextLimeExplainerInstance(explainer, labels, args))
  elif args['type'] == 'image' and args['algorithm'] == 'lime':
    instances.append(_ImageLimeExplainerInstance(explainer, labels, args))
  elif args['type'] == 'image' and args['algorithm'] == 'ig':
    instances.append(_ImageIgExplainerInstance(explainer, labels, args))
  elif args['type'] == 'tabular':
    instances.append(_TabularLimeExplainerInstance(explainer, labels, args))

  for i, label in enumerate(labels):
    IPython.display.display(
        IPython.display.HTML('<br/>Explaining features for label <b>"%s"</b><br/>' % label))
    for instance in instances:
      instance.visualize(i)


def _tensorboard_start(args, cell):
  datalab_ml.TensorBoard.start(args['logdir'])


def _tensorboard_stop(args, cell):
  datalab_ml.TensorBoard.stop(args['pid'])


def _tensorboard_list(args, cell):
  return datalab_ml.TensorBoard.list()


def _get_evaluation_csv_schema(csv_file):
  # ML Workbench produces predict_results_schema.json in local batch prediction.
  schema_file = os.path.join(os.path.dirname(csv_file), 'predict_results_schema.json')
  if not file_io.file_exists(schema_file):
    raise ValueError('csv data requires headers.')
  return schema_file


def _evaluate_cm(args, cell):
  if args['csv']:
    if args['headers']:
      headers = args['headers'].split(',')
      cm = datalab_ml.ConfusionMatrix.from_csv(args['csv'], headers=headers)
    else:
      schema_file = _get_evaluation_csv_schema(args['csv'])
      cm = datalab_ml.ConfusionMatrix.from_csv(args['csv'], schema_file=schema_file)
  elif args['bigquery']:
    cm = datalab_ml.ConfusionMatrix.from_bigquery(args['bigquery'])
  else:
    raise ValueError('Either csv or bigquery is needed.')

  if args['plot']:
    return cm.plot(figsize=(args['size'], args['size']), rotation=90)
  else:
    return cm.to_dataframe()


def _create_metrics(args):
  if args['csv']:
    if args['headers']:
      headers = args['headers'].split(',')
      metrics = datalab_ml.Metrics.from_csv(args['csv'], headers=headers)
    else:
      schema_file = _get_evaluation_csv_schema(args['csv'])
      metrics = datalab_ml.Metrics.from_csv(args['csv'], schema_file=schema_file)
  elif args['bigquery']:
    metrics = datalab_ml.Metrics.from_bigquery(args['bigquery'])
  else:
    raise ValueError('Either csv or bigquery is needed.')

  return metrics


def _evaluate_accuracy(args, cell):
  metrics = _create_metrics(args)
  return metrics.accuracy()


def _evaluate_regression(args, cell):
  metrics = _create_metrics(args)
  metrics_dict = []
  metrics_dict.append({
      'metric': 'Root Mean Square Error',
      'value': metrics.rmse()
  })
  metrics_dict.append({
      'metric': 'Mean Absolute Error',
      'value': metrics.mae()
  })
  metrics_dict.append({
      'metric': '50 Percentile Absolute Error',
      'value': metrics.percentile_nearest(50)
  })
  metrics_dict.append({
      'metric': '90 Percentile Absolute Error',
      'value': metrics.percentile_nearest(90)
  })
  metrics_dict.append({
      'metric': '99 Percentile Absolute Error',
      'value': metrics.percentile_nearest(99)
  })
  return pd.DataFrame(metrics_dict)


def _evaluate_pr(args, cell):
  metrics = _create_metrics(args)
  df = metrics.precision_recall(args['num_thresholds'], args['target_class'],
                                probability_column=args['probability_column'])
  if args['plot']:
    plt.plot(df['recall'], df['precision'],
             label='Precision-Recall curve for class ' + args['target_class'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.show()
  else:
    return df


def _evaluate_roc(args, cell):
  metrics = _create_metrics(args)
  df = metrics.roc(args['num_thresholds'], args['target_class'],
                   probability_column=args['probability_column'])
  if args['plot']:
    plt.plot(df['fpr'], df['tpr'],
             label='ROC curve for class ' + args['target_class'])
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('ROC')
    plt.legend(loc="lower left")
    plt.show()
  else:
    return df


def _model_list(args, cell):
  if args['name']:
    # model name provided. List versions of that model.
    versions = datalab_ml.ModelVersions(args['name'], project_id=args['project'])
    versions = list(versions.get_iterator())
    df = pd.DataFrame(versions)
    df['name'] = df['name'].apply(lambda x: x.split('/')[-1])
    df = df.replace(np.nan, '', regex=True)
    return df
  else:
    # List all models.
    models = list(datalab_ml.Models(project_id=args['project']).get_iterator())
    if len(models) > 0:
      df = pd.DataFrame(models)
      df['name'] = df['name'].apply(lambda x: x.split('/')[-1])
      df['defaultVersion'] = df['defaultVersion'].apply(lambda x: x['name'].split('/')[-1])
      df = df.replace(np.nan, '', regex=True)
      return df
    else:
      print('No models found.')


def _model_delete(args, cell):
  parts = args['name'].split('.')
  if len(parts) == 1:
    models = datalab_ml.Models(project_id=args['project'])
    models.delete(parts[0])
  elif len(parts) == 2:
    versions = datalab_ml.ModelVersions(parts[0], project_id=args['project'])
    versions.delete(parts[1])
  else:
    raise ValueError('Too many "." in name. Use "model" or "model.version".')


def _model_deploy(args, cell):
  parts = args['name'].split('.')
  if len(parts) == 2:
    model_name, version_name = parts[0], parts[1]
    model_exists = False
    try:
      # If describe() works, the model already exists.
      datalab_ml.Models(project_id=args['project']).get_model_details(model_name)
      model_exists = True
    except:
      pass

    if not model_exists:
      datalab_ml.Models(project_id=args['project']).create(model_name)

    versions = datalab_ml.ModelVersions(model_name, project_id=args['project'])
    runtime_version = args['runtime_version']
    if not runtime_version:
      runtime_version = tf.__version__
    versions.deploy(version_name, args['path'], runtime_version=runtime_version)
  else:
    raise ValueError('Name must be like "model.version".')


def _dataset_create(args, cell):
  if args['format'] == 'csv':
    if not args['schema']:
      raise ValueError('schema is required if format is csv.')

    schema, schema_file = None, None
    if isinstance(args['schema'], six.string_types):
      schema_file = args['schema']
    elif isinstance(args['schema'], list):
      schema = args['schema']
    else:
      raise ValueError('schema should either be a file path, or a dictionary.')

    train_dataset = datalab_ml.CsvDataSet(args['train'], schema=schema, schema_file=schema_file)
    eval_dataset = datalab_ml.CsvDataSet(args['eval'], schema=schema, schema_file=schema_file)
  elif args['format'] == 'bigquery':
    train_dataset = datalab_ml.BigQueryDataSet(table=args['train'])
    eval_dataset = datalab_ml.BigQueryDataSet(table=args['eval'])
  elif args['format'] == 'transformed':
    train_dataset = datalab_ml.TransformedDataSet(args['train'])
    eval_dataset = datalab_ml.TransformedDataSet(args['eval'])
  else:
    raise ValueError('Invalid data format.')

  dataset = DataSet(train_dataset, eval_dataset)
  google.datalab.utils.commands.notebook_environment()[args['name']] = dataset


def _dataset_explore(args, cell):

  dataset = get_dataset_from_arg(args['name'])
  print('train data instances: %d' % dataset.train.size)
  print('eval data instances: %d' % dataset.eval.size)

  if args['overview'] or args['facets']:
    if isinstance(dataset.train, datalab_ml.TransformedDataSet):
      raise ValueError('transformed data does not support overview or facets.')

    print('Sampled %s instances for each.' % args['sample_size'])
    sample_train_df = dataset.train.sample(args['sample_size'])
    sample_eval_df = dataset.eval.sample(args['sample_size'])
    if args['overview']:
      overview = datalab_ml.FacetsOverview().plot({'train': sample_train_df,
                                                   'eval': sample_eval_df})
      IPython.display.display(overview)
    if args['facets']:
      sample_train_df['_source'] = pd.Series(['train'] * len(sample_train_df),
                                             index=sample_train_df.index)
      sample_eval_df['_source'] = pd.Series(['eval'] * len(sample_eval_df),
                                            index=sample_eval_df.index)
      df_merged = pd.concat([sample_train_df, sample_eval_df])
      diveview = datalab_ml.FacetsDiveview().plot(df_merged)
      IPython.display.display(diveview)
