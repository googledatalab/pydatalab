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
import pandas as pd
import six


class FacetsOverview(object):
  """Represents A facets overview. """

  def plot(self, data):
    """ Plots an overview in a list of dataframes

    Args:
      data: a dictionary with key the name, and value the dataframe.
    """

    import IPython

    if not isinstance(data, dict) or not all(isinstance(v, pd.DataFrame) for v in data.values()):
      raise ValueError('Expect a dictionary with values all DataFrames.')

    gfsg = GenericFeatureStatisticsGenerator()
    data = [{'name': k, 'table': v} for k, v in six.iteritems(data)]
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

  def plot(self, data):
    """ Plots a detail view of data.

    Args:
      data: a Pandas dataframe.
    """

    import IPython

    if not isinstance(data, pd.DataFrame):
      raise ValueError('Expect a DataFrame.')

    jsonstr = data.to_json(orient='records')
    # Escape ' and " because the string will be in <script> block.
    jsonstr = jsonstr.replace('\'', '\\\'').replace('\\"', '\\\\\\"')
    html_id = 'f' + datalab.utils.commands.Html.next_id()
    HTML_TEMPLATE = """
        <link rel="import" href="/nbextensions/gcpdatalab/extern/facets-jupyter.html">
        <facets-dive id="{html_id}" height="600"></facets-dive>
        <script>
          var data = JSON.parse('{jsonstr}');
          document.querySelector("#{html_id}").data = data;
        </script>"""
    html = HTML_TEMPLATE.format(html_id=html_id, jsonstr=jsonstr)
    return IPython.core.display.HTML(html)
