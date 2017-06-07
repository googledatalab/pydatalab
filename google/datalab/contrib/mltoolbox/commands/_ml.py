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
import pandas as pd
import numpy as np
import six

import google.datalab.utils.commands
import google.datalab.contrib.mltoolbox._local_predict as _local_predict


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
                                   'Required if prediction data contains image columns.')
  predict_parser.add_argument('--no_show_image', action='store_true', default=False,
                              help='If not set, add a column of images in output.')

  predict_parser.set_defaults(func=_predict)

  return google.datalab.utils.commands.handle_magic_line(line, cell, parser)


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
