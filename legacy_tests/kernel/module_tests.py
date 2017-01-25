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
import mock
import sys
import unittest

# import Python so we can mock the parts we need to here.
import IPython
import IPython.core.magic


def noop_decorator(func):
  return func

IPython.core.magic.register_line_cell_magic = noop_decorator
IPython.core.magic.register_line_magic = noop_decorator
IPython.core.magic.register_cell_magic = noop_decorator
IPython.get_ipython = mock.Mock()

import datalab.utils.commands


class TestCases(unittest.TestCase):

  def test_create_python_module(self):
    datalab.utils.commands._modules._create_python_module('bar', 'y=1')
    self.assertIsNotNone(sys.modules['bar'])
    self.assertEqual(1, sys.modules['bar'].y)

  def test_pymodule(self):
    datalab.utils.commands._modules.pymodule('--name foo', 'x=1')
    self.assertIsNotNone(sys.modules['foo'])
    self.assertEqual(1, sys.modules['foo'].x)

  @mock.patch('datalab.utils.commands._modules._pymodule_cell', autospec=True)
  def test_pymodule_magic(self, mock_pymodule_cell):
    datalab.utils.commands._modules.pymodule('-n foo')
    mock_pymodule_cell.assert_called_with({
      'name': 'foo',
      'func': datalab.utils.commands._modules._pymodule_cell
    }, None)
