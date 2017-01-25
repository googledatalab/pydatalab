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


from plotly.offline import iplot


class ConfusionMatrix(object):
  """Represents a confusion matrix."""

  def __init__(self, predicted_labels, true_labels, counts):
    """Initializes an instance of a ComfusionMatrix. the length of predicted_values,
       true_values, count must be the same.

    Args:
      predicted_labels: a list of predicted labels.
      true_labels: a list of true labels.
      counts: a list of count for each (predicted, true) combination.

    Raises: Exception if predicted_labels, true_labels, and counts are not of the same size
    """
    if len(predicted_labels) != len(true_labels) or len(true_labels) != len(counts):
      raise Exception('The input predicted_labels, true_labels, counts need to be same size.')
    self._all_labels = list(set(predicted_labels) | set(true_labels))
    data = []
    for value in self._all_labels:
      predicts_for_current_true_label = \
          {p: c for p, t, c in zip(predicted_labels, true_labels, counts) if t == value}
      # sort by all_values and fill in zeros if needed
      predicts_for_current_true_label = [predicts_for_current_true_label.get(v, 0)
          for v in self._all_labels]
      data.append(predicts_for_current_true_label)
    self._data = data

  def plot(self):
    """Plot the confusion matrix."""
    figure_data = \
    {
      "data": [
        {
          "x": self._all_labels,
          "y": self._all_labels,
          "z": self._data,
          "colorscale": "YlGnBu",
          "type": "heatmap"
        }
      ],
      "layout": {
        "title": "Confusion Matrix",
        "xaxis": {
          "title": "Predicted value",
        },
        "yaxis": {
          "title": "True Value",
        }
      }
    }
    iplot(figure_data)
