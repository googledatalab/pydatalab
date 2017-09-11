# Copyright 2017 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Platform library - ML Workbench Model Prediction Explainer."""
from __future__ import absolute_import
from __future__ import unicode_literals

import csv
import numpy as np
from PIL import Image
import six
from tensorflow.python.lib.io import file_io

from . import _local_predict


class PredictionExplainer(object):
    """An explainer that explains text and image predictions based on LIME."""

    def __init__(self, model_dir):
        """
        Args:
          model_dir: the directory of the model to use for prediction.
        """

        self._model_dir = model_dir
        schema, features = _local_predict.get_model_schema_and_features(model_dir)
        self._headers = [x['name'] for x in schema]
        self._text_columns, self._image_columns = [], []
        for k, v in six.iteritems(features):
            if v['transform'] in ['image_to_vec']:
                self._image_columns.append(v['source_column'])
            elif v['transform'] in ['bag_of_words', 'tfidf']:
                self._text_columns.append(v['source_column'])

    def _make_text_predict_fn(self, labels, instance, column_to_explain):
        """Create a predict_fn that can be used by LIME text explainer. """

        def _predict_fn(perturbed_text):
            predict_input = []
            for x in perturbed_text:
                instance_copy = dict(instance)
                instance_copy[column_to_explain] = x
                predict_input.append(instance_copy)

            df = _local_predict.get_prediction_results(self._model_dir, predict_input,
                                                       self._headers, with_source=False)
            probs = _local_predict.get_probs_for_labels(labels, df)
            return np.asarray(probs)

        return _predict_fn

    def _make_image_predict_fn(self, labels, instance, column_to_explain):
        """Create a predict_fn that can be used by LIME image explainer. """

        def _predict_fn(perturbed_image):

            predict_input = []
            for x in perturbed_image:
                instance_copy = dict(instance)
                instance_copy[column_to_explain] = Image.fromarray(x)
                predict_input.append(instance_copy)

            df = _local_predict.get_prediction_results(
                self._model_dir, predict_input, self._headers,
                img_cols=self._image_columns, with_source=False)
            probs = _local_predict.get_probs_for_labels(labels, df)
            return np.asarray(probs)

        return _predict_fn

    def explain_text(self, labels, instance, column_name=None, num_features=10, num_samples=5000):
        """Explain a text field of a prediction.

        It analyze the prediction by LIME, and returns a report of which words are most impactful
        in contributing to certain labels.

        Args:
          labels: a list of labels to explain.
          instance: the prediction instance. It needs to conform to model's input. Can be a csv
              line string, or a dict.
          column_name: which text column to explain. Can be None if there is only one text column
              in the model input.
          num_features: maximum number of words (features) to analyze. Passed to
              LIME LimeTextExplainer directly.
          num_samples: size of the neighborhood to learn the linear model. Passed to
              LIME LimeTextExplainer directly.

        Returns:
          A LIME's LimeTextExplainer.

        Throws:
          ValueError if the given text column is not found in model input or column_name is None
              but there are multiple text columns in model input.
        """

        from lime.lime_text import LimeTextExplainer

        if len(self._text_columns) > 1 and not column_name:
            raise ValueError('There are multiple text columns in the input of the model. ' +
                             'Please specify "column_name".')
        elif column_name and column_name not in self._text_columns:
            raise ValueError('Specified column_name "%s" not found in the model input.'
                             % column_name)

        text_column_name = column_name if column_name else self._text_columns[0]
        if isinstance(instance, six.string_types):
            instance = csv.DictReader([instance], fieldnames=self._headers).next()

        predict_fn = self._make_text_predict_fn(labels, instance, text_column_name)
        explainer = LimeTextExplainer(class_names=labels)
        exp = explainer.explain_instance(
            instance[text_column_name], predict_fn, labels=range(len(labels)),
            num_features=num_features, num_samples=num_samples)
        return exp

    def explain_image(self, labels, instance, column_name=None, num_features=100000,
                      num_samples=300, batch_size=200, hide_color=0):
        """Explain an image of a prediction.

        It analyze the prediction by LIME, and returns a report of which words are most impactful
        in contributing to certain labels.

        Args:
          labels: a list of labels to explain.
          instance: the prediction instance. It needs to conform to model's input. Can be a csv
              line string, or a dict.
          column_name: which image column to explain. Can be None if there is only one image column
              in the model input.
          num_features: maximum number of areas (features) to analyze. Passed to
              LIME LimeImageExplainer directly.
          num_samples: size of the neighborhood to learn the linear model. Passed to
              LIME LimeImageExplainer directly.
          batch_size: size of batches passed to predict_fn. Passed to
              LIME LimeImageExplainer directly.
          hide_color: the color used to perturb images. Passed to
              LIME LimeImageExplainer directly.

        Returns:
          A LIME's LimeImageExplainer.

        Throws:
          ValueError if the given image column is not found in model input or column_name is None
              but there are multiple image columns in model input.
        """

        from lime.lime_image import LimeImageExplainer

        if len(self._image_columns) > 1 and not column_name:
            raise ValueError('There are multiple image columns in the input of the model. ' +
                             'Please specify "column_name".')
        elif column_name and column_name not in self._image_columns:
            raise ValueError('Specified column_name "%s" not found in the model input.'
                             % column_name)

        image_column_name = column_name if column_name else self._image_columns[0]
        if isinstance(instance, six.string_types):
            instance = csv.DictReader([instance], fieldnames=self._headers).next()

        predict_fn = self._make_image_predict_fn(labels, instance, image_column_name)
        explainer = LimeImageExplainer()
        with file_io.FileIO(instance[image_column_name], 'rb') as fi:
            im = Image.open(fi)
        im.thumbnail((299, 299), Image.ANTIALIAS)
        rgb_im = np.asarray(im.convert('RGB'))
        exp = explainer.explain_instance(
            rgb_im, predict_fn, labels=range(len(labels)), top_labels=None,
            hide_color=hide_color, num_features=num_features,
            num_samples=num_samples, batch_size=batch_size)
        return exp
