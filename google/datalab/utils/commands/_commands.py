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

"""Implementation of command parsing and handling within magics."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

try:
  import IPython
except ImportError:
  raise Exception('This module can only be loaded in ipython.')

import argparse
import shlex

import google.datalab.utils


class CommandParser(argparse.ArgumentParser):
  """An argument parser to parse commands in line/cell magic declarations. """

  def __init__(self, *args, **kwargs):
    """Initializes an instance of a CommandParser. """

    # It is important to initialize the members first before super init because
    # add_argument() is called during super's init, and add_argument() uses the
    # member variables.
    self._subcommands = None
    self._args_to_parse_from_cell = []

    # Used for caching in _get_args_to_parse_from_cell().
    self._subparser_args_to_parse_from_cell = None

    # Set _parser_class, so that subparsers added will also be of this type.
    self._parser_class = CommandParser
    super(CommandParser, self).__init__(*args, **kwargs)

  @staticmethod
  def create(name):
    """Creates a CommandParser for a specific magic. """
    return CommandParser(prog=name)

  def exit(self, status=0, message=None):
    """Overridden exit method to stop parsing without calling sys.exit(). """
    raise Exception(message)

  def format_usage(self):
    """Overridden usage generator to use the full help message. """
    return self.format_help()

  @staticmethod
  def create_args(line, namespace):
    """ Expand any meta-variable references in the argument list. """
    args = []
    # Using shlex.split handles quotes args and escape characters.
    for arg in shlex.split(line):
      if not arg:
         continue
      if arg[0] == '$':
        var_name = arg[1:]
        if var_name in namespace:
          args.append((namespace[var_name]))
        else:
          raise Exception('Undefined variable referenced in command line: %s' % arg)
      else:
        args.append(arg)
    return args

  def _get_args_to_parse_from_cell(self):

    def _get_subparsers(p):
      """Recursively get subparsers."""

      subparsers = [
        subparser
        for action in p._actions
        if isinstance(action, argparse._SubParsersAction)
        for _, subparser in action.choices.items()
      ]
      ret = subparsers
      for sp in subparsers:
        ret += _get_subparsers(sp)
      return ret

    if not self._subparser_args_to_parse_from_cell:
      subparsers = _get_subparsers(self)
      subcommand_name_to_args = {}
      for subparser in subparsers:
        name = subparser.prog.split()[-1]
        args_to_parse = subparser._args_to_parse_from_cell
        subcommand_name_to_args[name] = args_to_parse
      self._subparser_args_to_parse_from_cell = subcommand_name_to_args
    return self._subparser_args_to_parse_from_cell

  def parse(self, line, cell, namespace=None):
    """Parses a line and cell into a dictionary of arguments, expanding variables from a namespace.

    For each line parameters beginning with --, it also checks the cell content and see if it exists
    there. For example, if "--config1" is a line parameter, it checks to see if cell dict contains
    "config1" item, and if so, use the cell value. The "config1" item will also be removed from
    cell content.

    Args:
      line: line content.
      cell: cell content.
      namespace: user namespace. If None, IPython's user namespace is used.

    Returns:
      A tuple of: 1. parsed config dict. 2. remaining cell after line parameters are extracted.
    """

    if namespace is None:
      ipy = IPython.get_ipython()
      namespace = ipy.user_ns
    try:
      args = CommandParser.create_args(line, namespace)
      subparser_name_to_args = self._get_args_to_parse_from_cell()
      last_subcmd = next((x for x in reversed(args) if x in subparser_name_to_args), None)
      args_to_parse_from_cell = subparser_name_to_args[last_subcmd] if last_subcmd else None
      if args_to_parse_from_cell:
        cell_config = None
        try:
          cell_config, cell = google.datalab.utils.commands.parse_config_for_selected_keys(
              cell, namespace, args_to_parse_from_cell)
        except:
          # It is okay --- probably because cell is not in yaml or json format.
          pass

        if cell_config:
          for arg_name in cell_config:
            arg_value = cell_config[arg_name]
            if isinstance(arg_value, bool):
              if arg_value:
                line += ' --%s' % arg_name
            else:
              line += ' --%s %s' % (arg_name, str(cell_config[arg_name]))

      args = CommandParser.create_args(line, namespace)
      return self.parse_args(args), cell
    except Exception as e:
      print(str(e))
      return None, None

  def add_argument(self, *args, **kwargs):
    # Track add all line arguments beginning with '--'.
    for pos_arg in args:
      if pos_arg.startswith('--'):
        self._args_to_parse_from_cell.append(pos_arg[2:])

    return super(CommandParser, self).add_argument(*args, **kwargs)

  def subcommand(self, name, help, **kwargs):
    """Creates a parser for a sub-command. """
    if self._subcommands is None:
      self._subcommands = self.add_subparsers(help='commands')
    return self._subcommands.add_parser(name, description=help, help=help, **kwargs)
