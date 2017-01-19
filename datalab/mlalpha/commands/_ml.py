# Copyright 2017 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

try:
  import IPython
  import IPython.core.magic
except ImportError:
  raise Exception('This module can only be loaded in ipython.')

import collections
import google.cloud.ml as cloudml
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import yaml


import datalab.context
import datalab.mlalpha
import datalab.utils.commands


@IPython.core.magic.register_line_cell_magic
def ml(line, cell=None):
  """Implements the ml line cell magic.

  Args:
    line: the contents of the ml line.
    cell: the contents of the ml cell.

  Returns:
    The results of executing the cell.
  """
  parser = datalab.utils.commands.CommandParser(prog="ml", description="""
Execute various ml-related operations. Use "%%ml <command> -h" for help on a specific command.
""")
  preprocess_parser = parser.subcommand('preprocess', 'Run a preprocess job.')
  preprocess_parser.add_argument('--usage',
                                 help='Show usage from the specified preprocess package.',
                                 action='store_true', default=False)
  preprocess_parser.add_argument('--cloud',
                                 help='Whether to run the preprocessing job in the cloud.',
                                 action='store_true', default=False)
  preprocess_parser.add_argument('--package',
                                 help='The preprocess package to use. Can be a gs or local path.',
                                 required=True)
  preprocess_parser.set_defaults(func=_preprocess)

  train_parser = parser.subcommand('train', 'Train an ML model.')
  train_parser.add_argument('--usage',
                            help='Show usage from the specified trainer package',
                            action='store_true', default=False)
  train_parser.add_argument('--cloud',
                            help='Whether to run the training job in the cloud.',
                            action='store_true', default=False)
  train_parser.add_argument('--package',
                            help='The trainer package to use. Can be a gs or local path.',
                            required=True)
  train_parser.set_defaults(func=_train)

  predict_parser = parser.subcommand('predict', 'Predict with an ML model.')
  predict_parser.add_argument('--usage',
                              help='Show usage from the specified prediction package',
                              action='store_true', default=False)
  predict_parser.add_argument('--cloud',
                              help='Whether to run prediction in the cloud.',
                              action='store_true', default=False)
  predict_parser.add_argument('--package',
                              help='The prediction package to use. Can be a gs or local path.',
                              required=True)
  predict_parser.set_defaults(func=_predict)

  batch_predict_parser = parser.subcommand('batch_predict', 'Batch predict with an ML model.')
  batch_predict_parser.add_argument('--usage',
                                    help='Show usage from the specified prediction package',
                                    action='store_true', default=False)
  batch_predict_parser.add_argument('--cloud',
                                    help='Whether to run prediction in the cloud.',
                                    action='store_true', default=False)
  batch_predict_parser.add_argument('--package',
                                    help='The prediction package to use. Can be a gs or local path.',
                                    required=True)
  batch_predict_parser.set_defaults(func=_batch_predict)

  confusion_matrix_parser = parser.subcommand('confusion_matrix',
                                              'Plot confusion matrix. The source is provided ' +
                                              'in one of "csv", "bqtable", and "sql" params.')
  confusion_matrix_parser.add_argument('--csv',
                                       help='GCS or local path of CSV file which contains ' +
                                            '"target", "predicted" columns at least. The CSV ' +
                                            'either comes with a schema file in the same dir, ' +
                                            'or specify "headers: name1, name2..." in cell.')
  confusion_matrix_parser.add_argument('--bqtable',
                                       help='name of the BigQuery table in the form of ' + 
                                            'dataset.table.')
  confusion_matrix_parser.add_argument('--sql',
                                       help='name of the sql module defined in previous cell ' + 
                                            'which should return "target", "predicted", ' +
                                            'and "count" columns at least in results.')
  confusion_matrix_parser.set_defaults(func=_confusion_matrix)

  namespace = datalab.utils.commands.notebook_environment()
  return datalab.utils.commands.handle_magic_line(line, cell, parser, namespace=namespace)


def _command_template(pr, func_name):
  """Return (args_list, docstring).
     args_list is in the form of:
       arg1:
       arg2:
       arg3: (optional)
  """
  argspec, docstring = pr.get_func_args_and_docstring(func_name)
  num_defaults = len(argspec.defaults) if argspec.defaults is not None else 0
  # Need to fill in a keyword (here '(NOT_OP)') for non optional args.
  # Later we will replace '(NOT_OP)' with empty string.
  optionals = ['(NOT_OP)'] * (len(argspec.args) - num_defaults) + \
      ['(optional)'] * num_defaults
  args = dict(zip(argspec.args, optionals))
  args_dump = yaml.safe_dump(args, default_flow_style=False).replace('(NOT_OP)', '')
  return args_dump, docstring


def _run_package(args, cell, mode):
  local_func_name = 'local_' + mode
  cloud_func_name = 'cloud_' + mode
  with datalab.mlalpha.PackageRunner(args['package']) as pr:
    if args['usage'] is True:
      #TODO Consider calling _command_template once to save one pip installation
      command_local = """%%ml %s --package %s""" % (mode, args['package'])
      args_local, docstring_local = _command_template(pr, local_func_name)
      command_cloud = """%%ml %s --package %s --cloud""" % (mode, args['package'])
      args_cloud, docstring_cloud = _command_template(pr, cloud_func_name)
      output = """
Local Run Command:

%s
%s
[Description]:
%s

Cloud Run Command:

%s
%s
[Description]:
%s
""" % (command_local, args_local, docstring_local, command_cloud, args_cloud, docstring_cloud)
      return datalab.utils.commands.render_text(output, preformatted=True)

    env = datalab.utils.commands.notebook_environment()
    func_args = datalab.utils.commands.parse_config(cell, env)
    if args['cloud'] is True:
      return pr.run_func(cloud_func_name, func_args)
    else:
      return pr.run_func(local_func_name, func_args)


def _preprocess(args, cell):
  return _run_package(args, cell, 'preprocess')


def _train(args, cell):
  return _run_package(args, cell, 'train')


def _predict(args, cell):
  return _run_package(args, cell, 'predict')


def _batch_predict(args, cell):
  return _run_package(args, cell, 'batch_predict')


def _plot_confusion_matrix(cm, labels):
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title('Confusion matrix')
  plt.colorbar()
  tick_marks = np.arange(len(labels))
  plt.xticks(tick_marks, labels, rotation=45)
  plt.yticks(tick_marks, labels)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')


def _confusion_matrix_from_csv(input_csv, cell):
  schema_file = input_csv + '.schema.yaml'
  headers = None
  if cell is not None:
    env = datalab.utils.commands.notebook_environment()
    config = datalab.utils.commands.parse_config(cell, env)
    headers_str = config.get('headers', None)
    if headers_str is not None:
      headers = [x.strip() for x in headers_str.split(',')]
  if headers is not None:
    with cloudml.util._file.open_local_or_gcs(input_csv, mode='r') as f:
      df = pd.read_csv(f, names=headers)
  elif cloudml.util._file.file_exists(schema_file):
    df = datalab.mlalpha.csv_to_dataframe(input_csv, schema_file)
  else:
    raise Exception('headers is missing from cell, ' +
                    'and there is no schema file in the same dir as csv')
  labels = sorted(set(df['target']) | set(df['predicted']))
  cm = confusion_matrix(df['target'], df['predicted'], labels=labels)
  return cm, labels


def _confusion_matrix_from_query(sql_module_name, bq_table):
  if sql_module_name is not None:
    item = datalab.utils.commands.get_notebook_item(sql_module_name)
    query, _ = datalab.data.SqlModule.get_sql_statement_with_environment(item, {})
  else:
    query = ('select target, predicted, count(*) as count from %s group by target, predicted'
             % bq_table)
  dfbq = datalab.bigquery.Query(query).results().to_dataframe()
  labels = sorted(set(dfbq['target']) | set(dfbq['predicted']))
  labels_count = len(labels)
  dfbq['target'] = [labels.index(x) for x in dfbq['target']]
  dfbq['predicted'] = [labels.index(x) for x in dfbq['predicted']]
  cm = [[0]*labels_count for i in range(labels_count)]
  for index, row in dfbq.iterrows():
    cm[row['target']][row['predicted']] = row['count']
  return cm, labels


def _confusion_matrix(args, cell):
  if args['csv'] is not None:
    #TODO: Maybe add cloud run for large CSVs with federated table.
    cm, labels = _confusion_matrix_from_csv(args['csv'], cell)
  elif args['sql'] is not None or args['bqtable'] is not None:
    cm, labels = _confusion_matrix_from_query(args['sql'], args['bqtable'])
  else:
    raise Exception('One of "csv", "bqtable", and "sql" param is needed.')
  _plot_confusion_matrix(cm, labels)
