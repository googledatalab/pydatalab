# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Local implementation for preprocessing, training and prediction for inception model.
"""

import collections
import csv
import google.cloud.ml as ml
import itertools
import json
import os
import tensorflow as tf
import yaml


def _make_batches(iterable, batch_size):
  """Given an iterable, it creates another iterable with each element an iterable
     of a batch. In the last batch, since there are likely not enough elements for
     a whole batch, None is padded.
  """
  args = [iter(iterable)] * batch_size
  return itertools.izip_longest(fillvalue=None, *args)


def _expand_batches(results):
  """Given an iterable of batches (each batch is an iterable), create a "merged"
     iterable that each element is a single element in a batch.
  """
  for result_batch in results:
    for elem in result_batch:
      yield elem


def get_first(iterable):
  for item in iterable:
    yield item[0]


def _tf_predict(model_dir, batches):
  model_dir = os.path.join(model_dir, 'model')
  with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(os.path.join(model_dir, 'export.meta'))
    new_saver.restore(sess, os.path.join(model_dir, 'export'))
    init_op = tf.get_collection(tf.contrib.session_bundle.constants.INIT_OP_KEY)[0]
    sess.run(init_op)
    inputs = json.loads(tf.get_collection('inputs')[0])
    outputs = json.loads(tf.get_collection('outputs')[0])
    for batch in batches:
      feed_dict = collections.defaultdict(list)
      for ii, image_filename in enumerate(batch):
        if image_filename is None:
          break
        with ml.util._file.open_local_or_gcs(image_filename, 'r') as ff:
          image_bytes = ff.read()
          feed_dict[inputs['image_bytes']].append(image_bytes)
          feed_dict[inputs['key']].append(str(ii))
      predictions, labels, scores = sess.run(
          [outputs['prediction'], outputs['labels'], outputs['scores']], feed_dict=feed_dict)
      yield zip(predictions, labels, scores)


def predict(model_dir, image_files):
  """Local prediction."""

  # Single batch for instant prediction.
  results = next(_tf_predict(model_dir, [image_files]))
  predicted_and_scores = [(predicted, label_scores[list(labels).index(predicted)])
                          for predicted, labels, label_scores in results]
  return predicted_and_scores


def batch_predict(model_dir, input_csv, output_file, output_bq_table):
  """Local batch prediction."""

  input_csv_f = ml.util._file.read_file_stream(input_csv)
  reader = csv.reader(input_csv_f)
  images = _make_batches(get_first(reader), 20)
  
  results_batches = _tf_predict(model_dir, images)
  results = _expand_batches(results_batches)
  input_csv_f = ml.util._file.read_file_stream(input_csv)
  reader = csv.reader(input_csv_f)
  include_target = False
  with ml.util._file.open_local_or_gcs(output_file, mode='w') as f_out:
    writer = csv.writer(f_out)
    for input, prediction in zip(reader, results):
      predicted = prediction[0]
      labels = list(prediction[1])
      class_scores = prediction[2]
      if include_target or len(input) > 1:
        include_target = True
        target_index = labels.index(input[1])
        target_prob = class_scores[target_index]
      predicted_index = labels.index(predicted)
      predicted_prob = class_scores[predicted_index]
      target_prob_value = [str(target_prob)] if include_target else []
      writer.writerow(input + [predicted] + target_prob_value + [str(predicted_prob)] +
                      map(str, class_scores))
  if include_target:
    schema = [
        {'name': 'image_url', 'type': 'STRING'},
        {'name': 'target', 'type': 'STRING'},
        {'name': 'predicted', 'type': 'STRING'},
        {'name': 'target_prob', 'type': 'FLOAT'},
        {'name': 'predicted_prob', 'type': 'FLOAT'},
    ]
  else:
    schema = [
        {'name': 'image_url', 'type': 'STRING'},
        {'name': 'predicted', 'type': 'STRING'},
        {'name': 'predicted_prob', 'type': 'FLOAT'},
    ]
  for l in labels:
    schema.append({'name': 'prob_' + l, 'type': 'FLOAT'})
  schema_file = output_file + '.schema.yaml'
  with ml.util._file.open_local_or_gcs(schema_file, 'w') as yaml_file:
    yaml.dump(schema, yaml_file, default_flow_style=False)

  if output_bq_table is not None:
    import datalab.bigquery as bq
    dataset_name, table_name = output_bq_table.split('.')
    bq.Dataset(dataset_name).create()
    eval_results_table = bq.Table(output_bq_table).create(schema, overwrite = True)
    eval_results_table.load(output_file, mode='append', source_format='csv')
  return output_file, schema_file
