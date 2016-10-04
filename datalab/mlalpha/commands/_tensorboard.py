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

try:
  import IPython
  import IPython.core.magic
except ImportError:
  raise Exception('This module can only be loaded in ipython.')

import datalab.mlalpha
import datalab.utils.commands


@IPython.core.magic.register_line_cell_magic
def tensorboard(line, cell=None):
  """Implements the tensorboard cell magic.

  Args:
    line: the contents of the tensorboard line.
  Returns:
    The results of executing the cell.
  """
  parser = datalab.utils.commands.CommandParser(prog='tensorboard', description="""
Execute tensorboard operations. Use "%tensorboard <command> -h" for help on a specific command.
""")
  list_parser = parser.subcommand('list', 'List running TensorBoard instances.')
  list_parser.set_defaults(func=_list)
  start_parser = parser.subcommand('start', 'Start a TensorBoard server with the given logdir.')
  start_parser.add_argument('--logdir',
                            help='The directory containing TensorFlow events. ' +
                                 'Can be a GCS or local path.',
                            required=True)
  start_parser.set_defaults(func=_start)
  stop_parser = parser.subcommand('stop', 'Stop a TensorBoard server with the given pid.')
  stop_parser.add_argument('--pid',
                           help='The pid of the TensorBoard instance to stop.',
                           required=True)
  stop_parser.set_defaults(func=_stop)
  namespace = datalab.utils.commands.notebook_environment()
  return datalab.utils.commands.handle_magic_line(line, cell, parser, namespace=namespace)


def _list(args, _):
  """ List the running TensorBoard instances. """
  return datalab.utils.commands.render_dictionary(
             datalab.mlalpha.TensorBoardManager.get_running_list(),
             ['pid', 'logdir', 'port'])


def _start(args, _):
  """ Start a TensorBoard instance. """
  pid, port = datalab.mlalpha.TensorBoardManager.start(args['logdir'])
  url = datalab.mlalpha.TensorBoardManager.get_reverse_proxy_url(port)
  html = '<p>TensorBoard was started successfully with pid %d. ' % pid
  html += 'Click <a href="%s" target="_blank">here</a> to access it.</p>' % url
  return IPython.core.display.HTML(html)


def _stop(args, _):
  """ Stop a TensorBoard instance. """
  datalab.mlalpha.TensorBoardManager.stop(int(args['pid']))

