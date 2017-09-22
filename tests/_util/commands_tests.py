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

from __future__ import absolute_import
from __future__ import unicode_literals

import argparse
import contextlib
import six
import sys
import unittest
import yaml
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from google.datalab.utils.commands import CommandParser


class TestCases(unittest.TestCase):

  @staticmethod
  @contextlib.contextmanager
  # Redirects some stderr temporarily; can be used to prevent console output from some tests.
  def redirect_stderr(target):
    original = sys.stderr
    sys.stderr = target
    yield
    sys.stderr = original

  def test_subcommand_line(self):

    parser = CommandParser(
        prog='%test_subcommand_line',
        description='test_subcommand_line description')

    subcommand1 = parser.subcommand('subcommand1', help='subcommand1 help')
    subcommand1.add_argument('--string1', help='string1 help.')
    subcommand1.add_argument('--flag1', action='store_true', default=False,
                             help='flag1 help.')

    args, cell = parser.parse('subcommand1 --string1 value1 --flag1', None)
    self.assertEqual(args, {'string1': 'value1', 'flag1': True})
    self.assertIsNone(cell)

    args, cell = parser.parse('subcommand1 --string1 value1', None)
    self.assertEqual(args, {'string1': 'value1', 'flag1': False})
    self.assertIsNone(cell)

    args, cell = parser.parse('subcommand1', None)
    self.assertEqual(args, {'string1': None, 'flag1': False})
    self.assertIsNone(cell)

    # Adding same arg twice will cause argparse to raise its own ArgumentError.
    with self.assertRaises(argparse.ArgumentError):
      subcommand1.add_argument('--string2', help='string2 help.')
      subcommand1.add_argument('--string2', help='string2 help.')

  def test_subcommand_line_cell(self):

    parser = CommandParser(
        prog='%test_subcommand_line',
        description='test_subcommand_line description')

    subcommand1 = parser.subcommand('subcommand1', help='subcommand1 help')
    subcommands_of_subcommand1 = subcommand1.add_subparsers(dest='command')
    subcommand2 = subcommands_of_subcommand1.add_parser('subcommand2', help='subcommand2 help')
    subcommand2.add_argument('--string1', '-s', required=True, help='string1 help.')
    subcommand2.add_argument('--string2', '--string2again', dest='string2', help='string2 help.')
    subcommand2.add_cell_argument('string3', help='string3 help.')
    subcommand2.add_argument('--flag1', action='store_true', default=False,
                             help='flag1 help.')

    args, cell = parser.parse('subcommand1 subcommand2 -s value1 --string2 value2',
                              'flag1: true')
    self.assertEqual(args, {'string1': 'value1', 'string2': 'value2', 'string3': None,
                            'command': 'subcommand2', 'flag1': True})
    self.assertIsNone(cell)

    args, cell = parser.parse('subcommand1 subcommand2 --flag1',
                              'string1: value1\nstring2again: value2')
    self.assertEqual(args, {'string1': 'value1', 'string2': 'value2', 'string3': None,
                            'command': 'subcommand2', 'flag1': True})
    self.assertIsNone(cell)

    args, cell = parser.parse('subcommand1 subcommand2',
                              'string1: value1\nstring2: value2\nstring3: value3\n' +
                              'string4: value4\nflag1: false')
    self.assertEqual(args, {'string1': 'value1', 'string2': 'value2', 'string3': 'value3',
                            'command': 'subcommand2', 'flag1': False})
    self.assertEqual(yaml.load(cell), {'string3': 'value3', 'string4': 'value4'})

    # Regular arg and cell arg cannot be the same name.
    with self.assertRaises(ValueError):
      subcommand2.add_argument('--duparg', help='help.')
      subcommand2.add_cell_argument('duparg', help='help.')

    # Do not allow same arg in both line and cell.
    with self.assertRaises(ValueError):
      parser.parse('subcommand1 subcommand2 -s value1 --duparg v1', 'duparg: v2')

    # 'string3' is a cell arg. Argparse will raise Exception after finding an unrecognized param.
    with self.assertRaisesRegexp(Exception, 'unrecognized arguments: --string3 value3'):
      with TestCases.redirect_stderr(StringIO()):
        parser.parse('subcommand1 subcommand2 -s value1 --string3 value3', 'a: b')

    # 'string4' is required but missing.
    subcommand2.add_cell_argument('string4', required=True, help='string4 help.')
    with self.assertRaises(ValueError):
      parser.parse('subcommand1 subcommand2 -s value1', 'a: b')

  def test_subcommand_var_replacement(self):

    parser = CommandParser(
        prog='%test_subcommand_line',
        description='test_subcommand_line description')

    subcommand1 = parser.subcommand('subcommand1', help='subcommand1 help')
    subcommand1.add_argument('--string1', help='string1 help.')
    subcommand1.add_argument('--flag1', action='store_true', default=False,
                             help='flag1 help.')
    subcommand1.add_cell_argument('string2', help='string2 help.')
    subcommand1.add_cell_argument('dict1', help='dict1 help.')

    namespace = {'var1': 'value1', 'var2': 'value2', 'var3': [1, 2]}
    args, cell = parser.parse('subcommand1 --string1 $var1', 'a: b\nstring2: $var2', namespace)
    self.assertEqual(args,
                     {'string1': 'value1', 'string2': 'value2', 'flag1': False, 'dict1': None})
    self.assertEqual(yaml.load(cell), {'a': 'b', 'string2': '$var2'})

    cell = """
dict1:
  k1: $var1
  k2: $var3
"""
    args, cell = parser.parse('subcommand1', cell, namespace)
    self.assertEqual(args['dict1'], {'k1': 'value1', 'k2': [1, 2]})

  def test_subcommand_help(self):

    parser = CommandParser(
        prog='%test_subcommand_line',
        description='test_subcommand_line description')

    subcommand1 = parser.subcommand('subcommand1', help='subcommand1 help')
    subcommands_of_subcommand1 = subcommand1.add_subparsers(dest='command')
    subcommand2 = subcommands_of_subcommand1.add_parser('subcommand2', help='subcommand2 help')
    subcommand2.add_argument('--string1', '-s', required=True, help='string1 help.')
    subcommand2.add_argument('--string2', '--string2again', dest='string2', help='string2 help.')
    subcommand2.add_cell_argument('string3', help='string3 help.')
    subcommand2.add_argument('--flag1', action='store_true', default=False,
                             help='flag1 help.')

    old_stdout = sys.stdout
    buf = six.StringIO()
    sys.stdout = buf
    with self.assertRaises(Exception):
      parser.parse('subcommand1 subcommand2 --help', None)
    sys.stdout = old_stdout
    help_string = buf.getvalue()
    self.assertIn('string1 help.', help_string)
    self.assertIn('string2 help.', help_string)
    self.assertIn('string3 help.', help_string)
    self.assertIn('flag1 help.', help_string)
