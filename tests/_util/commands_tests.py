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
import unittest

from google.datalab.utils.commands import CommandParser


class TestCases(unittest.TestCase):

  def test_subcommand_line(self):

    parser = CommandParser(
        prog='%test_subcommand_line',
        description='test_subcommand_line description')

    subcommand1 = parser.subcommand('subcommand1', help='subcommand1 help')
    subcommand1.add_argument('--string1', help='string1 help.')
    subcommand1.add_argument('--flag1', action='store_true', default=False,
                             help='flag1 help.')

    args, cell = parser.parse('subcommand1 --string1 value1 --flag1', None)
    self.assertEqual(vars(args), {'string1': 'value1', 'flag1': True})
    self.assertIsNone(cell)

    args, cell = parser.parse('subcommand1 --string1 value1', 'string3: value3')
    self.assertEqual(vars(args), {'string1': 'value1', 'flag1': False})
    self.assertEqual(cell.strip(), 'string3: value3')

    args, cell = parser.parse('subcommand1', 'string1: value1\nflag1: true')
    self.assertEqual(vars(args), {'string1': 'value1', 'flag1': True})
    self.assertIsNone(cell)

    parser = CommandParser(
        prog='%test_subcommand_line',
        description='test_subcommand_line description')

    subcommand1 = parser.subcommand('subcommand1', help='subcommand1 help')
    subcommand1.add_argument('--string1', help='string1 help.')
    subcommand1.add_argument('--flag1', action='store_true', default=False,
                             help='flag1 help.')
    subcommands_of_subcommand1 = subcommand1.add_subparsers(dest='command')
    subcommand2 = subcommands_of_subcommand1.add_parser('subcommand2', help='subcommand2 help')
    subcommand2.add_argument('--string2', '-s', required=True, help='string2 help.')
    subcommand2.add_argument('--flag2', action='store_true', default=False,
                             help='flag2 help.')

    args, cell = parser.parse('subcommand1 subcommand2 -s value2', 'flag2: true')
    self.assertEqual(vars(args), {'string1': None, 'string2': 'value2',
                                  'flag1': False, 'command': 'subcommand2',
                                  'flag2': True})
    self.assertIsNone(cell)

    args, cell = parser.parse('subcommand1 subcommand2 --flag2', 'string2: value2')
    self.assertEqual(vars(args), {'string1': None, 'string2': 'value2',
                                  'flag1': False, 'command': 'subcommand2',
                                  'flag2': True})
    self.assertIsNone(cell)

    args, cell = parser.parse('subcommand1 subcommand2',
                              'string1: value1\nstring2: value2\nstring3: value3\nflag2: false')
    self.assertEqual(vars(args), {'string1': None, 'string2': 'value2',
                                  'flag1': False, 'command': 'subcommand2',
                                  'flag2': False})
    self.assertEqual(cell.strip(), 'string1: value1\nstring3: value3')
