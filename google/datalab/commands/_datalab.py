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

"""Google Cloud Platform library - datalab cell magic."""
from __future__ import absolute_import
from __future__ import unicode_literals

try:
  import IPython
  import IPython.core.display
  import IPython.core.magic
except ImportError:
  raise Exception('This module can only be loaded in ipython.')

import google.datalab.utils.commands


@IPython.core.magic.register_line_cell_magic
def datalab(line, cell=None):
  """Implements the datalab cell magic for ipython notebooks.

  Args:
    line: the contents of the datalab line.
  Returns:
    The results of executing the cell.
  """
  parser = google.datalab.utils.commands.CommandParser(
      prog='datalab',
      description="""
Execute operations that apply to multiple Datalab APIs.

Use "%datalab <command> -h" for help on a specific command.
""")

  config_parser = parser.subcommand(
      'config', help='List or set API-specific configurations.')
  config_sub_commands = config_parser.add_subparsers(dest='command')

  # %%datalab config list
  config_list_parser = config_sub_commands.add_parser(
      'list', help='List configurations')
  config_list_parser.set_defaults(func=_config_list_fn)

  # %%datalab config set -n <NAME> -v <VALUE>
  config_set_parser = config_sub_commands.add_parser(
      'set', help='Set configurations')
  config_set_parser.add_argument(
      '-n', '--name',
      help='The name of the configuration value', required=True)
  config_set_parser.add_argument(
      '-v', '--value', help='The value to set', required=True)
  config_set_parser.set_defaults(func=_config_set_fn)

  project_parser = parser.subcommand(
      'project', help='Get or set the default project ID')
  project_sub_commands = project_parser.add_subparsers(dest='command')

  # %%datalab project get
  project_get_parser = project_sub_commands.add_parser(
      'get', help='Get the default project ID')
  project_get_parser.set_defaults(func=_project_get_fn)

  # %%datalab project set -p <PROJECT_ID>
  project_set_parser = project_sub_commands.add_parser(
      'set', help='Set the default project ID')
  project_set_parser.add_argument(
      '-p', '--project', help='The default project ID', required=True)
  project_set_parser.set_defaults(func=_project_set_fn)

  return google.datalab.utils.commands.handle_magic_line(line, cell, parser)


def _config_list_fn(args, cell):
  ctx = google.datalab.Context.default()
  return google.datalab.utils.commands.render_dictionary([ctx.config])


def _config_set_fn(args, cell):
  name = args['name']
  value = args['value']
  ctx = google.datalab.Context.default()
  ctx.config[name] = value
  return google.datalab.utils.commands.render_dictionary([ctx.config])


def _project_get_fn(args, cell):
  ctx = google.datalab.Context.default()
  return google.datalab.utils.commands.render_text(ctx.project_id)


def _project_set_fn(args, cell):
  project = args['project']
  ctx = google.datalab.Context.default()
  ctx.set_project_id(project)
  return
