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

import json
import pandas as pd
from types import ModuleType

import google.datalab as datalab
import google.datalab.bigquery as bq


class FeatureSliceView(object):
  """Represents A feature slice view."""

  def _get_lantern_format(self, df):
    """ Feature slice view browser expects data in the format of:
          {"metricValues": {"count": 12, "accuracy": 1.0}, "feature": "species:Iris-setosa"}
          {"metricValues": {"count": 11, "accuracy": 0.72}, "feature": "species:Iris-versicolor"}
          ...
        This function converts a DataFrame to such format.
    """
    
    if ('count' not in df) or ('feature' not in df):
      raise Exception('No "count" or "feature" found in data.')
    if len(df.columns) < 3:
      raise Exception('Need at least one metrics column.')      
    if len(df) == 0:
      raise Exception('Data is empty')

    data = []
    for _, row in df.iterrows():
      metric_values = dict(row)
      feature = metric_values.pop('feature')
      data.append({'feature': feature, 'metricValues': metric_values})
    return data
  
  def plot(self, data):
    """ Plots a featire slice view on given data.

    Args:
      data: Can be one of:
            A string of sql query.
            A sql query module defined by "%%sql --module module_name".
            A pandas DataFrame.
          Regardless of data type, it must include the following columns:
            "feature": identifies a slice of features. For example: "petal_length:4.0-4.2".
            "count": number of instances in that slice of features.
          All other columns are viewed as metrics for its feature slice. At least one is required.
    """    
    import IPython

    if isinstance(data, basestring):
      data = bq.Query(data)

    if isinstance(data, bq.Query):
      df = data.execute().result().to_dataframe()
      data = self._get_lantern_format(df)
    elif isinstance(data, pd.core.frame.DataFrame):
      data = self._get_lantern_format(data)
    else:
      raise Exception('data needs to be a sql query, or a pandas DataFrame.')
      
    HTML_TEMPLATE = """<link rel="import" href="/nbextensions/gcpdatalab/extern/lantern-browser.html" >
        <lantern-browser id="{html_id}"></lantern-browser>
        <script>
        var browser = document.querySelector('#{html_id}');
        browser.metrics = {metrics};
        browser.data = {data};
        browser.sourceType = 'colab';
        browser.weightedExamplesColumn = 'count';
        browser.calibrationPlotUriFn = function(s) {{ return '/' + s; }}
        </script>"""
    # Serialize the data and list of metrics names to JSON string.
    metrics_str = str(map(str, data[0]['metricValues'].keys()))
    data_str = str([{str(k): json.dumps(v) for k,v in elem.iteritems()} for elem in data])
    html_id = 'l' + datalab.utils.commands.Html.next_id()
    html = HTML_TEMPLATE.format(html_id=html_id, metrics=metrics_str, data=data_str)
    IPython.display.display(IPython.display.HTML(html))

