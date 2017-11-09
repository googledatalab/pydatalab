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

import base64
import csv
import io
import numpy as np
import pandas as pd
from PIL import Image
import six
import tensorflow as tf
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
        self._categorical_columns, self._numeric_columns = [], []
        for k, v in six.iteritems(features):
            if v['transform'] in ['image_to_vec']:
                self._image_columns.append(v['source_column'])
            elif v['transform'] in ['bag_of_words', 'tfidf']:
                self._text_columns.append(v['source_column'])
            elif v['transform'] in ['one_hot', 'embedding']:
                self._categorical_columns.append(v['source_column'])
            elif v['transform'] in ['identity', 'scale']:
                self._numeric_columns.append(v['source_column'])

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

    def _get_unique_categories(self, df):
        """Get all categories for each categorical columns from training data."""

        categories = []
        for col in self._categorical_columns:
            categocial = pd.Categorical(df[col])
            col_categories = list(map(str, categocial.categories))
            col_categories.append('_UNKNOWN')
            categories.append(col_categories)
        return categories

    def _preprocess_data_for_tabular_explain(self, df, categories):
        """Get preprocessed training set in numpy array, and categorical names from raw training data.

        LIME tabular explainer requires a training set to know the distribution of numeric and
        categorical values. The training set has to be numpy arrays, with all categorical values
        converted to indices. It also requires list of names for each category.
        """

        df = df.copy()

        # Remove non tabular columns (text, image).
        for col in list(df.columns):
            if col not in (self._categorical_columns + self._numeric_columns):
                del df[col]

        # Convert categorical values into indices.
        for col_name, col_categories in zip(self._categorical_columns, categories):
            df[col_name] = df[col_name].apply(
                lambda x: col_categories.index(str(x)) if str(x) in col_categories
                else len(col_categories) - 1)

        # Make sure numeric values are really numeric
        for numeric_col in self._numeric_columns:
            df[numeric_col] = df[numeric_col].apply(lambda x: float(x))

        return df.as_matrix(self._categorical_columns + self._numeric_columns)

    def _make_tabular_predict_fn(self, labels, instance, categories):
        """Create a predict_fn that can be used by LIME tabular explainer. """

        def _predict_fn(np_instance):

            df = pd.DataFrame(
                np_instance,
                columns=(self._categorical_columns + self._numeric_columns))

            # Convert categorical indices back to categories.
            for col_name, col_categories in zip(self._categorical_columns, categories):
                df[col_name] = df[col_name].apply(lambda x: col_categories[int(x)])

            # Add columns that do not exist in the perturbed data,
            # such as key, text, and image data.
            for col_name in self._headers:
                if col_name not in (self._categorical_columns + self._numeric_columns):
                    df[col_name] = instance[col_name]

            r = _local_predict.get_prediction_results(
                self._model_dir, df, self._headers, with_source=False)
            probs = _local_predict.get_probs_for_labels(labels, r)
            probs = np.asarray(probs)
            return probs

        return _predict_fn

    def explain_tabular(self, trainset, labels, instance, num_features=5, kernel_width=3):
        """Explain categorical and numeric features for a prediction.

        It analyze the prediction by LIME, and returns a report of the most impactful tabular
        features contributing to certain labels.

        Args:
          trainset: a DataFrame representing the training features that LIME can use to decide
              value distributions.
          labels: a list of labels to explain.
          instance: the prediction instance. It needs to conform to model's input. Can be a csv
              line string, or a dict.
          num_features: maximum number of features to show.
          kernel_width: Passed to LIME LimeTabularExplainer directly.

        Returns:
          A LIME's lime.explanation.Explanation.
        """
        from lime.lime_tabular import LimeTabularExplainer

        if isinstance(instance, six.string_types):
            instance = next(csv.DictReader([instance], fieldnames=self._headers))

        categories = self._get_unique_categories(trainset)
        np_trainset = self._preprocess_data_for_tabular_explain(trainset, categories)
        predict_fn = self._make_tabular_predict_fn(labels, instance, categories)
        prediction_df = pd.DataFrame([instance])
        prediction_instance = self._preprocess_data_for_tabular_explain(prediction_df, categories)

        explainer = LimeTabularExplainer(
            np_trainset,
            feature_names=(self._categorical_columns + self._numeric_columns),
            class_names=labels,
            categorical_features=range(len(categories)),
            categorical_names={i: v for i, v in enumerate(categories)},
            kernel_width=kernel_width)

        exp = explainer.explain_instance(
            prediction_instance[0],
            predict_fn,
            num_features=num_features,
            labels=range(len(labels)))
        return exp

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
          A LIME's lime.explanation.Explanation.

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
            instance = next(csv.DictReader([instance], fieldnames=self._headers))

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
          A LIME's lime.explanation.Explanation.

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
            instance = next(csv.DictReader([instance], fieldnames=self._headers))

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

    def _image_gradients(self, input_csvlines, label, image_column_name):
        """Compute gradients from prob of label to image. Used by integrated gradients (probe)."""

        with tf.Graph().as_default() as g, tf.Session() as sess:
            logging_level = tf.logging.get_verbosity()
            try:
                tf.logging.set_verbosity(tf.logging.ERROR)
                meta_graph_pb = tf.saved_model.loader.load(
                    sess=sess,
                    tags=[tf.saved_model.tag_constants.SERVING],
                    export_dir=self._model_dir)
            finally:
                tf.logging.set_verbosity(logging_level)

            signature = meta_graph_pb.signature_def['serving_default']
            input_alias_map = {name: tensor_info_proto.name
                               for (name, tensor_info_proto) in signature.inputs.items()}
            output_alias_map = {name: tensor_info_proto.name
                                for (name, tensor_info_proto) in signature.outputs.items()}

            csv_tensor_name = list(input_alias_map.values())[0]

            # The image tensor is already built into ML Workbench graph.
            float_image = g.get_tensor_by_name("import/gradients_%s:0" % image_column_name)
            if label not in output_alias_map:
                raise ValueError('The label "%s" does not exist in output map.' % label)

            prob = g.get_tensor_by_name(output_alias_map[label])
            grads = tf.gradients(prob, float_image)[0]
            grads_values = sess.run(fetches=grads, feed_dict={csv_tensor_name: input_csvlines})

        return grads_values

    def probe_image(self, labels, instance, column_name=None, num_scaled_images=50,
                    top_percent=10):
        """ Get pixel importance of the image.

        It performs pixel sensitivity analysis by showing only the most important pixels to a
        certain label in the image. It uses integrated gradients to measure the
        importance of each pixel.

        Args:
            labels: labels to compute gradients from.
            instance: the prediction instance. It needs to conform to model's input. Can be a csv
              line string, or a dict.
            img_column_name: the name of the image column to probe. If there is only one image
                column it can be None.
            num_scaled_images: Number of scaled images to get grads from. For example, if 10,
                the image will be scaled by 0.1, 0.2, ..., 0,9, 1.0 and it will produce
                10 images for grads computation.
            top_percent: The percentile of pixels to show only. for example, if 10,
                only top 10% impactful pixels will be shown and rest of the pixels will be black.

        Returns:
            A tuple. First is the resized original image (299x299x3). Second is a list of
                the visualization with same size that highlights the most important pixels, one
                per each label.
        """

        if len(self._image_columns) > 1 and not column_name:
            raise ValueError('There are multiple image columns in the input of the model. ' +
                             'Please specify "column_name".')
        elif column_name and column_name not in self._image_columns:
            raise ValueError('Specified column_name "%s" not found in the model input.' %
                             column_name)

        image_column_name = column_name if column_name else self._image_columns[0]
        if isinstance(instance, six.string_types):
            instance = next(csv.DictReader([instance], fieldnames=self._headers))

        image_path = instance[image_column_name]

        with file_io.FileIO(image_path, 'rb') as fi:
            im = Image.open(fi)

        resized_image = im.resize((299, 299))

        # Produce a list of scaled images, create instances (csv lines) from these images.
        step = 1. / num_scaled_images
        scales = np.arange(0.0, 1.0, step) + step
        csv_lines = []
        for s in scales:
            pixels = (np.asarray(resized_image) * s).astype('uint8')
            scaled_image = Image.fromarray(pixels)
            buf = io.BytesIO()
            scaled_image.save(buf, "JPEG")
            encoded_image = base64.urlsafe_b64encode(buf.getvalue()).decode('ascii')
            instance_copy = dict(instance)
            instance_copy[image_column_name] = encoded_image

            buf = six.StringIO()
            writer = csv.DictWriter(buf, fieldnames=self._headers, lineterminator='')
            writer.writerow(instance_copy)
            csv_lines.append(buf.getvalue())

        integrated_gradients_images = []
        for label in labels:
          # Send to tf model to get gradients.
          grads = self._image_gradients(csv_lines, label, image_column_name)
          integrated_grads = np.average(grads, axis=0)

          # Gray scale the grads by removing color dimension.
          # abs() is for getting the most impactful pixels regardless positive or negative.
          grayed = np.average(abs(integrated_grads), axis=2)
          grayed = np.transpose([grayed, grayed, grayed], axes=[1, 2, 0])

          # Only show the most impactful pixels.
          p = np.percentile(grayed, 100 - top_percent)
          viz_window = np.where(grayed > p, 1, 0)
          vis = resized_image * viz_window
          im_vis = Image.fromarray(np.uint8(vis))
          integrated_gradients_images.append(im_vis)

        return resized_image, integrated_gradients_images
