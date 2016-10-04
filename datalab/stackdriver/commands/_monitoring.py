# Copyright 2016 Google Inc. All rights reserved.
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

"""IPython Functionality for the Google Monitoring API."""
from __future__ import absolute_import

try:
  import IPython.core.display
  import IPython.core.magic
except ImportError:
  raise Exception('This module can only be loaded in ipython.')

import datalab.stackdriver.monitoring as gcm
import datalab.utils.commands


@IPython.core.magic.register_line_cell_magic
def monitoring(line, cell=None):
  """Implements the monitoring cell magic for ipython notebooks.

  Args:
    line: the contents of the storage line.
  Returns:
    The results of executing the cell.
  """
  parser = datalab.utils.commands.CommandParser(prog='monitoring', description=(
      'Execute various Monitoring-related operations. Use "%monitoring '
      '<command> -h" for help on a specific command.'))

  list_parser = parser.subcommand(
      'list', 'List the metrics or resource types in a monitored project.')

  list_metric_parser = list_parser.subcommand(
      'metrics',
      'List the metrics that are available through the Monitoring API.')
  list_metric_parser.add_argument(
      '-t', '--type',
      help='The type of metric(s) to list; can include wildchars.')
  list_metric_parser.add_argument(
      '-p', '--project', help='The project on which to execute the request.')
  list_metric_parser.set_defaults(func=_list_metric_descriptors)

  list_resource_parser = list_parser.subcommand(
      'resource_types',
      ('List the monitored resource types that are available through the '
       'Monitoring API.'))
  list_resource_parser.add_argument(
      '-p', '--project', help='The project on which to execute the request.')
  list_resource_parser.add_argument(
      '-t', '--type',
      help='The resource type(s) to list; can include wildchars.')
  list_resource_parser.set_defaults(func=_list_resource_descriptors)

  list_group_parser = list_parser.subcommand(
      'groups',
      ('List the Stackdriver groups in this project.'))
  list_group_parser.add_argument(
      '-p', '--project', help='The project on which to execute the request.')
  list_group_parser.add_argument(
      '-n', '--name',
      help='The name of the group(s) to list; can include wildchars.')
  list_group_parser.set_defaults(func=_list_groups)

  return datalab.utils.commands.handle_magic_line(line, cell, parser)


def _list_metric_descriptors(args, _):
  """Lists the metric descriptors in the project."""
  project_id = args['project']
  pattern = args['type'] or '*'
  descriptors = gcm.MetricDescriptors(project_id=project_id)
  dataframe = descriptors.as_dataframe(pattern=pattern)
  return _render_dataframe(dataframe)


def _list_resource_descriptors(args, _):
  """Lists the resource descriptors in the project."""
  project_id = args['project']
  pattern = args['type'] or '*'
  descriptors = gcm.ResourceDescriptors(project_id=project_id)
  dataframe = descriptors.as_dataframe(pattern=pattern)
  return _render_dataframe(dataframe)


def _list_groups(args, _):
  """Lists the groups in the project."""
  project_id = args['project']
  pattern = args['name'] or '*'
  groups = gcm.Groups(project_id=project_id)
  dataframe = groups.as_dataframe(pattern=pattern)
  return _render_dataframe(dataframe)


def _render_dataframe(dataframe):
  """Helper to render a dataframe as an HTML table."""
  data = dataframe.to_dict(orient='records')
  fields = dataframe.columns.tolist()
  return IPython.core.display.HTML(
      datalab.utils.commands.HtmlBuilder.render_table(data, fields))
