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

# import Python so we can mock the parts we need to here.
import IPython.core.display
import IPython.core.magic


def noop_decorator(func):
  return func

IPython.core.magic.register_line_cell_magic = noop_decorator
IPython.core.magic.register_line_magic = noop_decorator
IPython.core.magic.register_cell_magic = noop_decorator
IPython.core.display.HTML = lambda x: x
IPython.core.display.JSON = lambda x: x

import datalab.utils.commands


class TestCases(unittest.TestCase):

  def test_chart_cell(self):
    t = [{'country': 'US', 'quantity': 100}, {'country': 'ZA', 'quantity': 50}]
    IPython.get_ipython().user_ns = {}
    chart = datalab.utils.commands._chart._chart_cell({'chart': 'geo', 'data': t, 'fields': None}, '')
    self.assertTrue(chart.find('charts.render(') > 0)
    self.assertTrue(chart.find('\'geo\'') > 0)
    self.assertTrue(chart.find('"fields": "*"') > 0)
    self.assertTrue(chart.find('{"c": [{"v": "US"}, {"v": 100}]}') > 0 or 
                    chart.find('{"c": [{"v": 100}, {"v": "US"}]}') > 0)
    self.assertTrue(chart.find('{"c": [{"v": "ZA"}, {"v": 50}]}') > 0 or
                    chart.find('{"c": [{"v": 50}, {"v": "ZA"}]}') > 0)


  def test_chart_magic(self):
    # TODO(gram): complete this test
    pass
