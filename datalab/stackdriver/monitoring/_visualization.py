# Copyright 2016 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.  See the License for the specific language governing permissions and limitations under
# the License.

"""Visualization methods."""

from __future__ import absolute_import


def table(dataframe, show_index=True):
  """Visualize a dataframe as an HTML table.

  Args:
    dataframe: the pandas dataframe to display.
    show_index: if False, the dataframe index is hidden.

  Returns:
    The HTML rendering of the dataframe as a table.
  """
  import IPython.core.display

  return IPython.core.display.HTML(dataframe.to_html(index=show_index))
