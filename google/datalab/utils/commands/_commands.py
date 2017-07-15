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
import six

import google.datalab.utils


class CommandParser(argparse.ArgumentParser):
  """An argument parser to parse commands in line/cell magic declarations. """

  def __init__(self, *args, **kwargs):
    """Initializes an instance of a CommandParser. """

    super(CommandParser, self).__init__(*args, **kwargs)
    # Set _parser_class, so that subparsers added will also be of this type.
    self._parser_class = CommandParser
    self._subcommands = None
    # A dict such as {'argname': {'required': True, 'help': 'arg help string'}}
    self._cell_args = {}

  @staticmethod
  def create(name):
    """Creates a CommandParser for a specific magic. """
    return CommandParser(prog=name)

  def exit(self, status=0, message=None):
    """Overridden exit method to stop parsing without calling sys.exit(). """
    if status == 0 and message is None:
      # This happens when parsing '--help'
      raise Exception('exit_0')
    else:
      raise Exception(message)

  def format_help(self):
    """Override help doc to add cell args. """

    if not self._cell_args:
      return super(CommandParser, self).format_help()
    else:
      # Print the standard argparse info, the cell arg block, and then the epilog
      # If we don't remove epilog before calling the super, then epilog will
      # be printed before the 'Cell args' block.
      epilog = self.epilog
      self.epilog = None
      orig_help = super(CommandParser, self).format_help()

      cell_args_help = '\nCell args:\n\n'
      for cell_arg, v in six.iteritems(self._cell_args):
        required = 'Required' if v['required'] else 'Optional'
        cell_args_help += '%s: %s. %s.\n\n' % (cell_arg, required, v['help'])

      orig_help += cell_args_help
      if epilog:
        orig_help += epilog + '\n\n'
      return orig_help

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

  def _get_subparsers(self):
    """Recursively get subparsers."""

    subparsers = []
    for action in self._actions:
      if isinstance(action, argparse._SubParsersAction):
        for _, subparser in action.choices.items():
          subparsers.append(subparser)

    ret = subparsers
    for sp in subparsers:
      ret += sp._get_subparsers()
    return ret

  def _get_subparser_line_args(self, subparser_prog):
    """ Get line args of a specified subparser by its prog."""

    subparsers = self._get_subparsers()
    for subparser in subparsers:
      if subparser_prog == subparser.prog:
        # Found the subparser.
        args_to_parse = []
        for action in subparser._actions:
          if action.option_strings:
            for argname in action.option_strings:
              if argname.startswith('--'):
                args_to_parse.append(argname[2:])
        return args_to_parse

    return None

  def _get_subparser_cell_args(self, subparser_prog):
    """ Get cell args of a specified subparser by its prog."""

    subparsers = self._get_subparsers()
    for subparser in subparsers:
      if subparser_prog == subparser.prog:
        return subparser._cell_args

    return None

  def add_cell_argument(self, name, help, required=False):
    """ Add a cell only argument.

    Args:
      name: name of the argument. No need to start with "-" or "--".
      help: the help string of the argument.
      required: Whether it is required in cell content.
    """

    for action in self._actions:
      if action.dest == name:
        raise ValueError('Arg "%s" was added by add_argument already.' % name)

    self._cell_args[name] = {'required': required, 'help': help}

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

    # Find which subcommand in the line by comparing line with subcommand progs.
    # For example, assuming there are 3 subcommands with their progs
    #   %bq tables
    #   %bq tables list
    #   %bq datasets
    # and the line is "tables list --dataset proj.myds"
    # it will find the second one --- "tables list" because it matches the prog and
    # it is the longest.
    args = CommandParser.create_args(line, namespace)

    # "prog" is a ArgumentParser's path splitted by namspace, such as '%bq tables list'.
    sub_parsers_progs = [x.prog for x in self._get_subparsers()]
    matched_progs = []
    for prog in sub_parsers_progs:
      # Remove the leading magic such as "%bq".
      match = prog.split()[1:]
      for i in range(len(args)):
        if args[i:i + len(match)] == match:
          matched_progs.append(prog)
          break

    matched_prog = None
    if matched_progs:
      # Get the longest match.
      matched_prog = max(matched_progs, key=lambda x: len(x.split()))

    # Line args can be provided in cell too. If they are in cell, move them to line
    # so we can parse them all together.
    line_args = self._get_subparser_line_args(matched_prog)
    if line_args:
      cell_config = None
      try:
        cell_config, cell = google.datalab.utils.commands.parse_config_for_selected_keys(
            cell, line_args)
      except:
        # It is okay --- probably because cell is not in yaml or json format.
        pass

      if cell_config:
        google.datalab.utils.commands.replace_vars(cell_config, namespace)
        for arg_name in cell_config:
          arg_value = cell_config[arg_name]
          if arg_value is None:
            continue

          if '--' + arg_name in args:
            raise ValueError('config item "%s" is specified in both cell and line.' % arg_name)
          if isinstance(arg_value, bool):
            if arg_value:
              line += ' --%s' % arg_name
          else:
            line += ' --%s %s' % (arg_name, str(cell_config[arg_name]))

    # Parse args again with the new line.
    args = CommandParser.create_args(line, namespace)
    args = vars(self.parse_args(args))

    # Parse cell args.
    cell_config = None
    cell_args = self._get_subparser_cell_args(matched_prog)
    if cell_args:
      try:
        cell_config, _ = google.datalab.utils.commands.parse_config_for_selected_keys(
            cell, cell_args)
      except:
        # It is okay --- probably because cell is not in yaml or json format.
        pass

      if cell_config:
        google.datalab.utils.commands.replace_vars(cell_config, namespace)

      for arg in cell_args:
        if (cell_args[arg]['required'] and
           (cell_config is None or cell_config.get(arg, None) is None)):
          raise ValueError('Cell config "%s" is required.' % arg)

    if cell_config:
      args.update(cell_config)

    return args, cell

  def subcommand(self, name, help, **kwargs):
    """Creates a parser for a sub-command. """
    if self._subcommands is None:
      self._subcommands = self.add_subparsers(help='commands')
    return self._subcommands.add_parser(name, description=help, help=help, **kwargs)
