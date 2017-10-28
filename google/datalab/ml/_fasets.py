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


import base64
import google.datalab as datalab
from google.datalab.utils.facets.generic_feature_statistics_generator \
    import GenericFeatureStatisticsGenerator
import numpy as np
import pandas as pd
import re
import six


class FacetsOverview(object):
  """Represents A facets overview. """

  def _remove_nonascii(self, df):
    """Make copy and remove non-ascii characters from it."""

    df_copy = df.copy(deep=True)
    for col in df_copy.columns:
      if (df_copy[col].dtype == np.dtype('O')):
        df_copy[col] = df[col].apply(
          lambda x: re.sub(r'[^\x00-\x7f]', r'', x) if isinstance(x, six.string_types) else x)

    return df_copy

  def plot(self, data):
    """ Plots an overview in a list of dataframes

    Args:
      data: a dictionary with key the name, and value the dataframe.
    """

    import IPython

    if not isinstance(data, dict) or not all(isinstance(v, pd.DataFrame) for v in data.values()):
      raise ValueError('Expect a dictionary where the values are all dataframes.')

    gfsg = GenericFeatureStatisticsGenerator()
    data = [{'name': k, 'table': self._remove_nonascii(v)} for k, v in six.iteritems(data)]
    data_proto = gfsg.ProtoFromDataFrames(data)
    protostr = base64.b64encode(data_proto.SerializeToString()).decode("utf-8")
    html_id = 'f' + datalab.utils.commands.Html.next_id()

    HTML_TEMPLATE = """<link rel="import" href="/nbextensions/gcpdatalab/extern/facets-jupyter.html" >
        <facets-overview id="{html_id}"></facets-overview>
        <script>
          document.querySelector("#{html_id}").protoInput = "{protostr}";
        </script>"""
    html = HTML_TEMPLATE.format(html_id=html_id, protostr=protostr)
    return IPython.core.display.HTML(html)


class FacetsDiveview(object):
  """Represents A facets overview. """

  def plot(self, data, height=1000, render_large_data=False):
    """ Plots a detail view of data.

    Args:
      data: a Pandas dataframe.
      height: the height of the output.
    """

    import IPython

    if not isinstance(data, pd.DataFrame):
      raise ValueError('Expect a DataFrame.')

    if (len(data) > 10000 and not render_large_data):
      raise ValueError('Facets dive may not work well with more than 10000 rows. ' +
                       'Reduce data or set "render_large_data" to True.')

    jsonstr = data.to_json(orient='records')
    html_id = 'f' + datalab.utils.commands.Html.next_id()
    HTML_TEMPLATE = """
        <link rel="import" href="/nbextensions/gcpdatalab/extern/facets-jupyter.html">
        <facets-dive id="{html_id}" height="{height}"></facets-dive>
        <script>
          var data = {jsonstr};
          document.querySelector("#{html_id}").data = data;
        </script>"""
    html = HTML_TEMPLATE.format(html_id=html_id, jsonstr=jsonstr, height=height)
    return IPython.core.display.HTML(html)
