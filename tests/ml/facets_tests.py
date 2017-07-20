# Copyright 2017 Google Inc. All rights reserved.
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

import pandas as pd

from google.datalab.ml import FacetsOverview, FacetsDiveview


class TestFacets(unittest.TestCase):
  """Tests facets visualization components."""

  def _create_test_data(self):
    data1 = [
      {'num1': 1.2, 'weekday': 'Monday', 'occupation': 'software engineer'},
      {'num1': 3.2, 'weekday': 'Tuesday', 'occupation': 'medical doctor'},
    ]

    data2 = [
      {'num1': -2.8, 'weekday': 'Friday', 'occupation': 'musician'},
    ]

    data1 = pd.DataFrame(data1)
    data2 = pd.DataFrame(data2)
    return data1, data2

  def test_overview_plot(self):
    """Tests overview."""

    data1, data2 = self._create_test_data()
    output = FacetsOverview().plot({'data1': data1, 'data2': data2})
    # Output is an html. Ideally we can parse the html and verify nodes, but since the html
    # is output by a polymer component which is tested separately, we just verify
    # minumum keywords.
    self.assertIn("facets-overview", output)
    self.assertIn("<script>", output)

  def test_dive_plot(self):
    """Tests diveview."""

    data1, _ = self._create_test_data()
    output = FacetsDiveview().plot(data1)

    # Output is an html. Ideally we can parse the html and verify nodes, but since the html
    # is output by a polymer component which is tested separately, we just verify
    # minumum keywords.
    self.assertIn("facets-dive", output)
    self.assertIn("<script>", output)
