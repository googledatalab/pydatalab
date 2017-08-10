# Copyright 2015 Google Inc. All rights reserved.
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

"""Google Cloud Platform library - Pipeline IPython Functionality."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

try:
  import IPython
  import IPython.core.display
  import IPython.core.magic
except ImportError:
  raise Exception('This module can only be deployed in ipython.')

import google.datalab.data
import google.datalab.utils
import google.datalab.utils.commands


def _create_cell(args, cell_body):
  """Implements the pipeline cell create magic used to create Pipeline objects.

  The supported syntax is:

      %%pipeline create <args>
      [<inline YAML>]

  Args:
    args: the arguments following '%%pipeline create'.
    cell_body: the contents of the cell
  """
  name = args.get('name')
  if name is None:
    raise Exception("Pipeline name was not specified.")

  pipeline = google.datalab.pipeline.Pipeline(cell_body, name,
                                              env=IPython.get_ipython().user_ns)
  google.datalab.utils.commands.notebook_environment()[name] = pipeline

  # TODO(rajivpb): Temporarily, print spec. Eventually, print better stuff
  return print(pipeline.py)


def _deploy_line(args):
  """Implements the pipeline deploy magic used to deploy pipelines in 
  airflow.

  The supported syntax is:

      %%pipeline deploy <args>

  Args:
    args: the optional arguments following '%%pipeline deploy'.
    cell_body: the contents of the cell
  """
  name = args['name']
  pipeline = google.datalab.utils.commands.notebook_environment()[name]

  # TODO(rajivpb): Temporarily, print. Eventually, copy to GCS
  return print(pipeline.py)


def _create_create_subparser(parser):
  create_parser = parser.subcommand('create', 'Create and/or execute a '
                                              'Pipeline object. If a pipeline '
                                              'name is not specified, the '
                                              'pipeline is scheduled.')
  create_parser.add_argument('-n', '--name', help='The name of this Pipeline '
                                                  'object')

  return create_parser


def _create_deploy_subparser(parser):
  deploy_parser = parser.subcommand('deploy', 'Load a pipeline object for '
                                              'scheduling.')
  deploy_parser.add_argument('-n', '--name', help='The name of the pipeline '
                                                  'object')

  return deploy_parser


def _add_command(
    parser, subparser_fn, handler, cell_required=False, cell_prohibited=False):
  """ Create and initialize a pipeline subcommand handler. """
  sub_parser = subparser_fn(parser)
  sub_parser.set_defaults(func=lambda args, cell: _dispatch_handler(
      args, cell, sub_parser, handler, cell_required=cell_required,
      cell_prohibited=cell_prohibited))


def _create_pipeline_parser():
  """ Create the parser for the %pipeline magics.

  Note that because we use the func default handler dispatch mechanism of 
  argparse, our handlers can take only one argument which is the parsed args. So
  we must create closures for the handlers that bind the cell contents and thus
  must recreate this parser for each cell upon execution.
  """
  parser = google.datalab.utils.commands.CommandParser(
      prog='%pipeline', description="""
Execute various pipeline-related operations. Use "%pipeline <command> -h"
for help on a specific command.
  """)

  # %%pipeline create
  _add_command(parser, _create_create_subparser, _create_cell)

  # %pipeline deploy
  _add_command(parser, _create_deploy_subparser, _deploy_line,
               cell_prohibited=True)

  return parser


_pipeline_parser = _create_pipeline_parser()


@IPython.core.magic.register_line_cell_magic
def pipeline(line, cell=None):
  """Implements the pipeline cell magic for ipython notebooks.

  The supported syntax is:

    %%pipeline <command> [<args>]
    <cell>

  or:

    %pipeline <command> [<args>]

  Use %pipeline --help for a list of commands, or %pipeline <command> --help for
  help on a specific command.
  """
  return google.datalab.utils.commands.handle_magic_line(line, cell,
                                                         _pipeline_parser)

def _dispatch_handler(
    args, cell, parser, handler, cell_required=False, cell_prohibited=False):
  """ Makes sure cell magics include cell and line magics don't, before 
    dispatching to handler.

  Args:
    args: the parsed arguments from the magic line.
    cell: the contents of the cell, if any.
    parser: the argument parser for <cmd>; used for error message.
    handler: the handler to call if the cell present/absent check passes.
    cell_required: True for cell magics, False for line magics that can't be 
    cell magics.
    cell_prohibited: True for line magics, False for cell magics that can't be 
    line magics.
  Returns:
    The result of calling the handler.
  Raises:
    Exception if the invocation is not valid.
  """
  if cell_prohibited:
    if cell and len(cell.strip()):
      parser.print_help()
      raise Exception(
          'Additional data is not supported with the %s command.' % parser.prog)
    return handler(args)

  if cell_required and not cell:
    parser.print_help()
    raise Exception('The %s command requires additional data' % parser.prog)

  return handler(args, cell)



def _repr_html_pipeline(pipeline):
  return google.datalab.utils.commands.HtmlBuilder.render_text(
      pipeline.py, preformatted=True)


def _register_html_formatters():
  try:
    ipy = IPython.get_ipython()
    html_formatter = ipy.display_formatter.formatters['text/html']

    html_formatter.for_type_by_name(
        'google.datalab.pipeline._pipeline', 'Pipeline', _repr_html_pipeline)

  except TypeError:
    # For when running unit tests
    pass


_register_html_formatters()