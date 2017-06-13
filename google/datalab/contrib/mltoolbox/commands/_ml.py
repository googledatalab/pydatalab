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
import tempfile

import google.datalab.utils.commands
import google.datalab.contrib.mltoolbox._local_predict as _local_predict
import google.datalab.contrib.mltoolbox._shell_process as _shell_process


MLTOOLBOX_CODE_PATH = '/datalab/lib/pydatalab/solutionbox/code_free_ml/mltoolbox/code_free_ml/'


@IPython.core.magic.register_line_cell_magic
def ml(line, cell=None):
  """Implements the datalab cell magic for MLToolbox operations.

  Args:
    line: the contents of the ml command line.
  Returns:
    The results of executing the cell.
  """
  parser = google.datalab.utils.commands.CommandParser(
      prog='%ml',
      description="""
Execute MLToolbox operations.

Use "%ml <command> -h" for help on a specific command.
""")

  analyze_parser = parser.subcommand(
      'analyze', help="""Analyze training data and generate stats, such as min/max/mean
for numeric values, vocabulary for text columns. Example usage:

%%ml analyze --output_dir path/to/dir [--cloud]
training_data:
  csv_file_pattern: path/to/csv
  csv_schema:
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

Cell input is in yaml format. Fields:

training_data: one of the following:
  csv_file_pattern and csv_schema (as the example above), or
  bigquery (example: "bigquery: project.dataset.table" or
                     "bigquery: select * from table where num1 > 1.0"), or
  a variable defined as google.datalab.ml.CsvDataSet or google.datalab.ml.BigQueryDataSet

features: A dictionary with key being column name. The list of supported transforms:
            "transform: identity"
                does nothing (for numerical columns).
            "transform: scale
             value: x"
                scale a numerical column to [-a, a]. If value is missing, x defaults to 1.
            "transform: one_hot"
                treats the string column as categorical and makes one-hot encoding of it.
            "transform: embedding
             embedding_dim: d"
                treats the string column as categorical and makes embeddings of it with specified
                dimension size.
            "transform: bag_of_words"
                treats the string column as text and make bag of words transform of it.
            "transform: tfidf"
                treats the string column as text and make TFIDF transform of it.
            "transform: image_to_vec"
                from image gs url to embeddings.
            "transform: target"
                denotes the column is the target. If the schema type of this column is string,
                a one_hot encoding is automatically applied. If numerical, an identity transform
                is automatically applied.
            "transform: key"
                column contains metadata-like information and will be output as-is in prediction.

Also support in-notebook variables, such as:
%%ml analyze --output_dir path/to/dir
training_data: $my_csv_dataset
features: $features_def
""", formatter_class=argparse.RawTextHelpFormatter)
  analyze_parser.add_argument('--output_dir', required=True,
                              help='path of output directory.')
  analyze_parser.add_argument('--cloud', action='store_true', default=False,
                              help='whether to run analysis in cloud or local.')

  analyze_parser.set_defaults(func=_analyze)

  predict_parser = parser.subcommand(
      'predict', help="""Predict with local or deployed models.
      'Prediction data by CSV lines in input cell in yaml format. For example:

%%ml predict --headers key,num --model path/to/model
prediction_data:
  - key1,value1
  - key2,value2

or define your data as a list of dict, or a list of CSV lines, or a pandas DataFrame.
For example, in another cell, define a list of dict:

my_data = [{'key': 1, 'num': 1.2}, {'key': 2, 'num': 2.8}]

Then:

%%ml predict --headers key,num --model path/to/model
prediction_data: $my_data
""", formatter_class=argparse.RawTextHelpFormatter)
  predict_parser.add_argument('--model', required=True,
                              help='The model path if not --cloud, or the id in ' +
                                   'the form of model.version if --cloud.')
  predict_parser.add_argument('--headers', required=True,
                              help='The comma seperated headers of the prediction data.')
  predict_parser.add_argument('--image_columns',
                              help='Comma seperated headers of image URL columns. ' +
                                   'Required if prediction data contains image URL columns.')
  predict_parser.add_argument('--no_show_image', action='store_true', default=False,
                              help='If not set, add a column of images in output.')

  predict_parser.set_defaults(func=_predict)

  return google.datalab.utils.commands.handle_magic_line(line, cell, parser)


def _analyze(args, cell):
  env = google.datalab.utils.commands.notebook_environment()
  cell_data = google.datalab.utils.commands.parse_config(cell, env)
  google.datalab.utils.commands.validate_config(cell_data,
                                                required_keys=['training_data', 'features'])
  # For now, always run python2. If needed we can run python3 when the current kernel
  # is py3. Since now our transform cannot work on py3 anyway, I would rather run
  # everything with python2.
  cmd_args = ['python', 'analyze.py', '--output-dir', args['output_dir']]
  if args['cloud']:
    cmd_args.append('--cloud')

  training_data = cell_data['training_data']
  tmpdir = tempfile.mkdtemp()

  def _create_json_file(data, filename):
    json_file = os.path.join(tmpdir, filename)
    with open(json_file, 'w') as f:
      json.dump(data, f)
    return json_file

  try:
    if isinstance(training_data, dict):
      if 'csv_file_pattern' in training_data and 'csv_schema' in training_data:
        schema = training_data['csv_schema']
        schema_file = _create_json_file(schema, 'schema.json')
        cmd_args.extend(['--csv-file-pattern', training_data['csv_file_pattern']])
        cmd_args.extend(['--csv-schema-file', schema_file])
      elif 'bigquery' in training_data:
        cmd_args.extend(['--bigquery-table', training_data['bigquery']])
      else:
        raise ValueError('Invalid training_data dict. ' +
                         'Requires either "csv_file_pattern" and "csv_schema", or "bigquery".')
    elif isinstance(training_data, google.datalab.ml.CsvDataSet):
      schema_file = _create_json_file(training_data.schema, 'schema.json')
      # TODO: Modify when command line interface supports multiple csv files.
      cmd_args.extend(['--csv-file-pattern', training_data.input_files[0]])
      cmd_args.extend(['--csv-schema-file', schema_file])
    elif isinstance(training_data, google.datalab.ml.BigQueryDataSet):
      # TODO: Support query too once command line supports query.
      cmd_args.extend(['--bigquery-table', training_data.table])
    else:
      raise ValueError('Invalid training data. Requires either a dict, ' +
                       'a google.datalab.ml.CsvDataSet, or a google.datalab.ml.BigQueryDataSet.')

    features = cell_data['features']
    features_file = _create_json_file(features, 'features.json')
    cmd_args.extend(['--features-file', features_file])
    _shell_process.run_and_monitor(cmd_args, os.getpid(), cwd=MLTOOLBOX_CODE_PATH)
  finally:
    shutil.rmtree(tmpdir)


def _predict(args, cell):
  headers = args['headers'].split(',')
  img_cols = args['image_columns'].split(',') if args['image_columns'] else []
  env = google.datalab.utils.commands.notebook_environment()

  cell_data = google.datalab.utils.commands.parse_config(cell, env)
  if 'prediction_data' not in cell_data:
    raise ValueError('Missing "prediction_data" in cell.')

  data = cell_data['prediction_data']
  df = _local_predict.get_prediction_results(args['model'], data, headers,
                                             img_cols, not args['no_show_image'])

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
