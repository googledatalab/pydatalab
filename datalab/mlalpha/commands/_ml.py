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
import numpy as np
import os
import pandas as pd
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
