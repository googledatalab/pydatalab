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

"""Google Cloud Platform library - ml cell magic."""
from __future__ import absolute_import
from __future__ import unicode_literals


import base64
import collections
import copy
import csv
from io import BytesIO
import json
import logging
import numpy as np
import os
import pandas as pd
from PIL import Image
import six
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import signature_constants

import google.datalab.ml as ml


def _tf_load_model(sess, model_dir):
  """Load a tf model from model_dir, and return input/output alias maps."""

  meta_graph_pb = tf.saved_model.loader.load(
      sess=sess,
      tags=[tf.saved_model.tag_constants.SERVING],
      export_dir=model_dir)

  signature = meta_graph_pb.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  input_alias_map = {friendly_name: tensor_info_proto.name
                     for (friendly_name, tensor_info_proto) in signature.inputs.items()}
  output_alias_map = {friendly_name: tensor_info_proto.name
                      for (friendly_name, tensor_info_proto) in signature.outputs.items()}

  return input_alias_map, output_alias_map


def _tf_predict(model_dir, input_csvlines):
  """Prediction with a tf savedmodel.

  Args:
    model_dir: directory that contains a saved model
    input_csvlines: list of csv strings

  Returns:
    Dict in the form tensor_name:prediction_list. Note that the value is always
        a list, even if there was only 1 row in input_csvlines.
  """

  with tf.Graph().as_default(), tf.Session() as sess:
    input_alias_map, output_alias_map = _tf_load_model(sess, model_dir)
    csv_tensor_name = list(input_alias_map.values())[0]
    results = sess.run(fetches=output_alias_map,
                       feed_dict={csv_tensor_name: input_csvlines})

  # convert any scalar values to a list. This may happen when there is one
  # example in input_csvlines and the model uses tf.squeeze on the output
  # tensor.
  if len(input_csvlines) == 1:
    for k, v in six.iteritems(results):
      if not isinstance(v, (list, np.ndarray)):
        results[k] = [v]

  # Convert bytes to string. In python3 the results may be bytes.
  for k, v in six.iteritems(results):
    if any(isinstance(x, bytes) for x in v):
      results[k] = [x.decode('utf-8') for x in v]

  return results


def _download_images(data, img_cols):
  """Download images given image columns."""

  images = collections.defaultdict(list)
  for d in data:
    for img_col in img_cols:
      if d.get(img_col, None):
        if isinstance(d[img_col], Image.Image):
          # If it is already an Image, just copy and continue.
          images[img_col].append(d[img_col])
        else:
          # Otherwise it is image url. Load the image.
          with file_io.FileIO(d[img_col], 'rb') as fi:
            im = Image.open(fi)
          images[img_col].append(im)
      else:
        images[img_col].append('')

  return images


def _get_predicton_csv_lines(data, headers, images):
  """Create CSV lines from list-of-dict data."""

  if images:
    data = copy.deepcopy(data)
    for img_col in images:
      for d, im in zip(data, images[img_col]):
        if im == '':
          continue

        im = im.copy()
        im.thumbnail((299, 299), Image.ANTIALIAS)
        buf = BytesIO()
        im.save(buf, "JPEG")
        content = base64.urlsafe_b64encode(buf.getvalue()).decode('ascii')
        d[img_col] = content

  csv_lines = []
  for d in data:
    buf = six.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers, lineterminator='')
    writer.writerow(d)
    csv_lines.append(buf.getvalue())

  return csv_lines


def _get_display_data_with_images(data, images):
  """Create display data by converting image urls to base64 strings."""

  if not images:
    return data

  display_data = copy.deepcopy(data)
  for img_col in images:
    for d, im in zip(display_data, images[img_col]):
      if im == '':
        d[img_col + '_image'] = ''
      else:
        im = im.copy()
        im.thumbnail((128, 128), Image.ANTIALIAS)
        buf = BytesIO()
        im.save(buf, "PNG")
        content = base64.b64encode(buf.getvalue()).decode('ascii')
        d[img_col + '_image'] = content

  return display_data


def get_model_schema_and_features(model_dir):
  """Get a local model's schema and features config.

  Args:
    model_dir: local or GCS path of a model.
  Returns:
    A tuple of schema (list) and features config (dict).
  """
  schema_file = os.path.join(model_dir, 'assets.extra', 'schema.json')
  schema = json.loads(file_io.read_file_to_string(schema_file))
  features_file = os.path.join(model_dir, 'assets.extra', 'features.json')
  features_config = json.loads(file_io.read_file_to_string(features_file))
  return schema, features_config


def get_prediction_results(model_dir_or_id, data, headers, img_cols=None,
                           cloud=False, with_source=True, show_image=True):
  """ Predict with a specified model.

  It predicts with the model, join source data with prediction results, and formats
  the results so they can be displayed nicely in Datalab.

  Args:
    model_dir_or_id: The model directory if cloud is False, or model.version if cloud is True.
    data: Can be a list of dictionaries, a list of csv lines, or a Pandas DataFrame. If it is not
        a list of csv lines, data will be converted to csv lines first, using the orders specified
        by headers and then send to model. For images, it can be image gs urls or in-memory PIL
        images. Images will be converted to base64 encoded strings before prediction.
    headers: the column names of data. It specifies the order of the columns when
        serializing to csv lines for prediction.
    img_cols: The image url columns. If specified, the img_urls will be converted to
        base64 encoded image bytes.
    with_source: Whether return a joined prediction source and prediction results, or prediction
        results only.
    show_image: When displaying prediction source, whether to add a column of image bytes for
        each image url column.

  Returns:
    A dataframe of joined prediction source and prediction results, or prediction results only.
  """

  if img_cols is None:
    img_cols = []

  if isinstance(data, pd.DataFrame):
    data = list(data.T.to_dict().values())
  elif isinstance(data[0], six.string_types):
    data = list(csv.DictReader(data, fieldnames=headers))

  images = _download_images(data, img_cols)
  predict_data = _get_predicton_csv_lines(data, headers, images)

  if cloud:
    parts = model_dir_or_id.split('.')
    if len(parts) != 2:
      raise ValueError('Invalid model name for cloud prediction. Use "model.version".')

    predict_results = ml.ModelVersions(parts[0]).predict(parts[1], predict_data)
  else:
    tf_logging_level = logging.getLogger("tensorflow").level
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    try:
      predict_results = _tf_predict(model_dir_or_id, predict_data)
    finally:
      logging.getLogger("tensorflow").setLevel(tf_logging_level)

  df_r = pd.DataFrame(predict_results)
  if not with_source:
    return df_r

  display_data = data
  if show_image:
    display_data = _get_display_data_with_images(data, images)

  df_s = pd.DataFrame(display_data)
  df = pd.concat([df_r, df_s], axis=1)
  # Remove duplicate columns. All 'key' columns are duplicate here.
  df = df.loc[:, ~df.columns.duplicated()]

  return df


def get_probs_for_labels(labels, prediction_results):
  """ Given ML Workbench prediction results, get probs of each label for each instance.

  The prediction results are like:
  [
    {'predicted': 'daisy', 'probability': 0.8, 'predicted_2': 'rose', 'probability_2': 0.1},
    {'predicted': 'sunflower', 'probability': 0.9, 'predicted_2': 'daisy', 'probability_2': 0.01},
    ...
  ]

  Each instance is ordered by prob. But in some cases probs are needed for fixed
  order of labels. For example, given labels = ['daisy', 'rose', 'sunflower'], the
  results of above is expected to be:
  [
    [0.8, 0.1, 0.0],
    [0.01, 0.0, 0.9],
    ...
  ]
  Note that the sum of each instance may not be always 1. If model's top_n is set to
  none-zero, and is less than number of labels, then prediction results may not contain
  probs for all labels.

  Args:
    labels: a list of labels specifying the order of the labels.
    prediction_results: a pandas DataFrame containing prediction results, usually returned
        by get_prediction_results() call.

  Returns:
    A list of list of probs for each class.
  """

  probs = []
  if 'probability' in prediction_results:
    # 'probability' exists so top-n is set to none zero, and results are like
    # "predicted, predicted_2,...,probability,probability_2,...
    for i, r in prediction_results.iterrows():
      probs_one = [0.0] * len(labels)
      for k, v in six.iteritems(r):
        if v in labels and k.startswith('predicted'):
          if k == 'predict':
            prob_name = 'probability'
          else:
            prob_name = 'probability' + k[9:]
          probs_one[labels.index(v)] = r[prob_name]
      probs.append(probs_one)
    return probs
  else:
    # 'probability' does not exist, so top-n is set to zero. Results are like
    # "predicted, class_name1, class_name2,...
    for i, r in prediction_results.iterrows():
      probs_one = [0.0] * len(labels)
      for k, v in six.iteritems(r):
        if k in labels:
          probs_one[labels.index(k)] = v
      probs.append(probs_one)
    return probs


def _batch_csv_reader(csv_file, n):
  with file_io.FileIO(csv_file, 'r') as f:
    args = [f] * n
    return six.moves.zip_longest(*args)


def _get_output_schema(session, output_alias_map):
  schema = []
  for name in sorted(six.iterkeys(output_alias_map)):
    tensor_name = output_alias_map[name]
    dtype = session.graph.get_tensor_by_name(tensor_name).dtype
    if dtype == tf.int32 or dtype == tf.int64:
      schema.append({'name': name, 'type': 'INTEGER'})
    elif dtype == tf.float32 or dtype == tf.float64:
      schema.append({'name': name, 'type': 'FLOAT'})
    else:
      schema.append({'name': name, 'type': 'STRING'})

  return schema


def _format_results(output_format, output_schema, batched_results):
  # Convert a dict of list to a list of dict.
  # Note that results from session.run may contain scaler value instead of lists
  # if batch size is 1.
  if (isinstance(next(iter(batched_results.values())), list) or
     isinstance(next(iter(batched_results.values())), np.ndarray)):
    batched_results = [dict(zip(batched_results, t)) for t in zip(*batched_results.values())]
  else:
    batched_results = [batched_results]

  if output_format == 'csv':
    results = []
    for r in batched_results:
      values = [str(r[schema['name']]) for schema in output_schema]
      results.append(','.join(values))
  elif output_format == 'json':
    # Default json encoder cannot handle numpy types.
    class _JSONEncoder(json.JSONEncoder):

      def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(_JSONEncoder, self).default(obj)

    results = [json.dumps(x, cls=_JSONEncoder) for x in batched_results]
  else:
    raise ValueError('Unknown output_format %s' % output_format)

  return results


def local_batch_predict(model_dir, csv_file_pattern, output_dir, output_format, batch_size=100):
  """ Batch Predict with a specified model.

  It does batch prediction, saves results to output files and also creates an output
  schema file. The output file names are input file names prepended by 'predict_results_'.

  Args:
    model_dir: The model directory containing a SavedModel (usually saved_model.pb).
    csv_file_pattern: a pattern of csv files as batch prediction source.
    output_dir: the path of the output directory.
    output_format: csv or json.
    batch_size: Larger batch_size improves performance but may
        cause more memory usage.
  """

  file_io.recursive_create_dir(output_dir)
  csv_files = file_io.get_matching_files(csv_file_pattern)
  if len(csv_files) == 0:
    raise ValueError('No files found given ' + csv_file_pattern)

  with tf.Graph().as_default(), tf.Session() as sess:
    input_alias_map, output_alias_map = _tf_load_model(sess, model_dir)
    csv_tensor_name = list(input_alias_map.values())[0]
    output_schema = _get_output_schema(sess, output_alias_map)
    for csv_file in csv_files:
      output_file = os.path.join(
          output_dir,
          'predict_results_' +
          os.path.splitext(os.path.basename(csv_file))[0] + '.' + output_format)
      with file_io.FileIO(output_file, 'w') as f:
        prediction_source = _batch_csv_reader(csv_file, batch_size)
        for batch in prediction_source:
          batch = [l.rstrip() for l in batch if l]
          predict_results = sess.run(fetches=output_alias_map, feed_dict={csv_tensor_name: batch})
          formatted_results = _format_results(output_format, output_schema, predict_results)
          f.write('\n'.join(formatted_results) + '\n')

  file_io.write_string_to_file(os.path.join(output_dir, 'predict_results_schema.json'),
                               json.dumps(output_schema, indent=2))
