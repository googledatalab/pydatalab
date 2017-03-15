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

import google.datalab
import google.datalab.stackdriver.monitoring as gcm
import google.datalab.utils.commands


@IPython.core.magic.register_line_cell_magic
def sd(line, cell=None):
  """Implements the stackdriver cell magic for ipython notebooks.

  Args:
    line: the contents of the storage line.
  Returns:
    The results of executing the cell.
  """
  parser = google.datalab.utils.commands.CommandParser(prog='%sd', description=(
      'Execute various Stackdriver related operations. Use "%sd '
      '<stackdriver_product> -h" for help on a specific Stackdriver product.'))

  # %%sd monitoring
  _create_monitoring_subparser(parser)
  return google.datalab.utils.commands.handle_magic_line(line, cell, parser)


def _create_monitoring_subparser(parser):
  monitoring_parser = parser.subcommand(
      'monitoring', 'Execute Stackdriver monitoring related operations. Use '
      '"sd monitoring <command> -h" for help on a specific command')

  metric_parser = monitoring_parser.subcommand(
      'metrics', 'Operations on Stackdriver Monitoring metrics')
  metric_list_parser = metric_parser.subcommand('list', 'List metrics')
  metric_list_parser.add_argument(
      '-p', '--project',
      help='The project whose metrics should be listed.')
  metric_list_parser.add_argument(
      '-t', '--type',
      help='The type of metric(s) to list; can include wildchars.')
  metric_list_parser.set_defaults(func=_monitoring_metrics_list)

  resource_parser = monitoring_parser.subcommand(
      'resource_types', 'Operations on Stackdriver Monitoring resource types')
  resource_list_parser = resource_parser.subcommand('list', 'List resource types')
  resource_list_parser.add_argument(
      '-p', '--project',
      help='The project whose resource types should be listed.')
  resource_list_parser.add_argument(
      '-t', '--type',
      help='The resource type(s) to list; can include wildchars.')
  resource_list_parser.set_defaults(func=_monitoring_resource_types_list)

  group_parser = monitoring_parser.subcommand(
      'groups', 'Operations on Stackdriver groups')
  group_list_parser = group_parser.subcommand('list', 'List groups')
  group_list_parser.add_argument(
      '-p', '--project',
      help='The project whose groups should be listed.')
  group_list_parser.add_argument(
      '-n', '--name',
      help='The name of the group(s) to list; can include wildchars.')
  group_list_parser.set_defaults(func=_monitoring_groups_list)


def _monitoring_metrics_list(args, _):
  """Lists the metric descriptors in the project."""
  project_id = args['project']
  pattern = args['type'] or '*'
  descriptors = gcm.MetricDescriptors(context=_make_context(project_id))
  dataframe = descriptors.as_dataframe(pattern=pattern)
  return _render_dataframe(dataframe)


def _monitoring_resource_types_list(args, _):
  """Lists the resource descriptors in the project."""
  project_id = args['project']
  pattern = args['type'] or '*'
  descriptors = gcm.ResourceDescriptors(context=_make_context(project_id))
  dataframe = descriptors.as_dataframe(pattern=pattern)
  return _render_dataframe(dataframe)


def _monitoring_groups_list(args, _):
  """Lists the groups in the project."""
  project_id = args['project']
  pattern = args['name'] or '*'
  groups = gcm.Groups(context=_make_context(project_id))
  dataframe = groups.as_dataframe(pattern=pattern)
  return _render_dataframe(dataframe)


def _render_dataframe(dataframe):
  """Helper to render a dataframe as an HTML table."""
  data = dataframe.to_dict(orient='records')
  fields = dataframe.columns.tolist()
  return IPython.core.display.HTML(
      google.datalab.utils.commands.HtmlBuilder.render_table(data, fields))


def _make_context(project_id):
  default_context = google.datalab.Context.default()
  if project_id:
    return google.datalab.Context(project_id, default_context.credentials)
  else:
    return default_context
