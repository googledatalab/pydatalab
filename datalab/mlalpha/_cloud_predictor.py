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
import json
from numbers import Number
import pandas as pd

import google.cloud.ml as ml

import datalab.context
import datalab.utils

from . import _metadata


# TODO(qimingj) Remove once the API is public since it will no longer be needed
_CLOUDML_DISCOVERY_URL = 'https://storage.googleapis.com/cloud-ml/discovery/' \
                         'ml_v1beta1_discovery.json'


class CloudPredictor(object):
  """Preforms cloud predictions on given data."""

  def __init__(self, model_name, version_name, metadata_path=None, label_output=None,
               project_id=None, credentials=None, api=None):
    """Initializes an instance of a CloudPredictor.

    Args:
      model_name: the name of the model used for prediction.
      version_name: the name of the version used for prediction.
      metadata_path: metadata that will be used to preprocess the instance data. If None,
          the instance data has to be preprocessed.
      label_output: the name of the output column where all values should be converted from
          index to labels. Only useful in classification. If specified, metadata_path is required.
      project_id: project_id of the model. If not provided, default project_id will be used.
      credentials: credentials used to talk to CloudML service. If not provided, default
          credentials will be used.
      api: an optional CloudML API client.
    """
    self._model_name = model_name
    self._version_name = version_name
    self._metadata_path = metadata_path
    self._metadata = None
    if metadata_path is not None:
      self._metadata = _metadata.Metadata(metadata_path)
    self._label_output = label_output
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

    if self._metadata_path is not None:
      transformer = ml.features.FeatureProducer(self._metadata_path)
      instances = [transformer.preprocess(i) for i in data]
    else:
      instances = [json.dumps(i) for i in data]
    request = self._api.projects().predict(body={'instances': instances},
                                           name=self._full_version_name)
    result = request.execute()
    if 'predictions' not in result:
      raise Exception('Invalid response from service. Cannot find "predictions" in response.')
    predictions = []
    for row in result['predictions']:
      prediction = json.loads(row)
      if (self._metadata is not None and self._label_output is not None
          and self._label_output in prediction):
        if not isinstance(prediction[self._label_output], Number):
            raise Exception('Cannot get labels because output "%s" is type %s but not number.'
                % (self._label_output, type(prediction[self._label_output])))
        label_index = prediction[self._label_output]
        prediction[self._label_output] = \
            str(self._metadata.get_classification_label(label_index)) + (' (%d)' % label_index)
      predictions.append(prediction)
    return predictions
