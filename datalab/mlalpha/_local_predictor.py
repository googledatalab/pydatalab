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


import collections
import json
from numbers import Number
import numpy
import os
import pandas as pd

import google.cloud.ml as ml

from . import _metadata


class LocalPredictor(object):
  """Preforms local predictions on given data.
  """

  def __init__(self, model_dir, label_output=None):
    """Initializes an instance of LocalPredictor.

    Args:
      model_dir: a directory that contains model checkpoint and metagraph. Can be local or GCS.
      label_output: the name of the output column where all values should be converted from
          index to labels. Only useful in classification. If specified, a metadata.yaml file is required.
    """
    self._model_dir = model_dir
    self._metadata_path = None
    self._metadata = None
    metadata_path = os.path.join(model_dir, "metadata.yaml")
    if ml.util._file.file_exists(metadata_path):
      self._metadata_path = metadata_path
      self._metadata = _metadata.Metadata(metadata_path)
    self._label_output = label_output

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

    Raises: Exception if the prediction result has incorrect label types
    """
    session, _ = ml.session_bundle.load_session_bundle_from_path(self._model_dir)
    # get the mappings between aliases and tensor names for both inputs and outputs
    input_key = json.loads(session.graph.get_collection(
        ml.session_bundle.INPUTS_KEY)[0]).values()[0]
    output_alias_map = json.loads(session.graph.get_collection(ml.session_bundle.OUTPUTS_KEY)[0])
    aliases, tensornames = zip(*output_alias_map.items())

    if isinstance(data, pd.DataFrame):
      data = data.T.to_dict().values()

    feed_dict = collections.defaultdict(list)
    if self._metadata_path is not None:
      transformer = ml.features.FeatureProducer(self._metadata_path)
      for instance in data:
        preprocessed = transformer.preprocess(instance)
        feed_dict[input_key].append(preprocessed.SerializeToString())
    else:
      for instance in data:
        feed_dict[input_key].append(json.dumps(instance))

    result = session.run(fetches=tensornames, feed_dict=feed_dict)
    if len(result) == 1:
      result = [result]
    predictions = []
    for row in zip(*result):
      prediction_row = {}
      for name, value in zip(aliases, row):
        if (self._metadata is not None and self._label_output is not None
            and name == self._label_output):
          if not isinstance(value, Number):
            raise Exception('Cannot get labels because output "%s" is type %s but not number.'
                % (name, type(value)))
          prediction_row[name] = str(self._metadata.get_classification_label(value)) + \
              (' (%d)' % value)
        elif isinstance(value, numpy.generic):
          prediction_row[name] = numpy.asscalar(value)
        elif isinstance(value, numpy.ndarray):
          prediction_row[name] = value.tolist()
        else:
          prediction_row[name] = value
      predictions.append(prediction_row)
    return predictions
