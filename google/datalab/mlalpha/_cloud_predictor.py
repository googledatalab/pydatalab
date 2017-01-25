# Copyright 2016 Google Inc. All rights reserved.
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


from googleapiclient import discovery
import pandas as pd

import datalab.context
import datalab.utils


# TODO(qimingj) Remove once the API is public since it will no longer be needed
_CLOUDML_DISCOVERY_URL = 'https://storage.googleapis.com/cloud-ml/discovery/' \
                         'ml_v1beta1_discovery.json'


class CloudPredictor(object):
  """Preforms cloud predictions on given data."""

  # TODO: Either remove label_output, or add code to load metadata from model dir and
  # transform integer to label. Depending on whether online prediction returns label or not.
  def __init__(self, model_name, version_name, label_output=None,
               project_id=None, credentials=None, api=None):
    """Initializes an instance of a CloudPredictor.

    Args:
      model_name: the name of the model used for prediction.
      version_name: the name of the version used for prediction.
      label_output: the name of the output column where all values should be converted from
          index to labels. Only useful in classification. If specified, metadata_path is required.
      project_id: project_id of the model. If not provided, default project_id will be used.
      credentials: credentials used to talk to CloudML service. If not provided, default
          credentials will be used.
      api: an optional CloudML API client.
    """
    self._model_name = model_name
    self._version_name = version_name
    if project_id is None:
      project_id = datalab.context.Context.default().project_id
    self._project_id = project_id
    if credentials is None:
      credentials = datalab.context.Context.default().credentials
    self._credentials = credentials
    if api is None:
      api = discovery.build('ml', 'v1beta1', credentials=self._credentials,
                            discoveryServiceUrl=_CLOUDML_DISCOVERY_URL)
    self._api = api
    self._full_version_name = ('projects/%s/models/%s/versions/%s' %
        (self._project_id, self._model_name, self._version_name))

  def predict(self, data):
    """Make predictions on given data.

    Args:
      data: a list of feature data or a pandas DataFrame. Each element in the list is an instance
          which is a dictionary of feature data.
          An example:
            [{"sepal_length": 4.9, "sepal_width": 2.5, "petal_length": 4.5, "petal_width": 1.7},
             {"sepal_length": 5.7, "sepal_width": 2.8, "petal_length": 4.1, "petal_width": 1.3}]
    Returns:
      A list of prediction results for given instances. Each element is a dictionary representing
          output mapping from the graph.
      An example:
        [{"predictions": 1, "score": [0.00078, 0.71406, 0.28515]},
         {"predictions": 1, "score": [0.00244, 0.99634, 0.00121]}]

    Raises: Exception if bad response is received from the service
            Exception if the prediction result has incorrect label types
    """
    if isinstance(data, pd.DataFrame):
      data = data.T.to_dict().values()

    request = self._api.projects().predict(body={'instances': data},
                                           name=self._full_version_name)
    request.headers['user-agent'] = 'GoogleCloudDataLab/1.0'
    result = request.execute()
    if 'predictions' not in result:
      raise Exception('Invalid response from service. Cannot find "predictions" in response.')

    return result['predictions']
