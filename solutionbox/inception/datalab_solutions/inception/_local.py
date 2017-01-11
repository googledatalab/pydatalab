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

import apache_beam as beam
import collections
import csv
import datetime
import google.cloud.ml as ml
import json
import os
import tensorflow as tf
import yaml

from . import _preprocess
from . import _model
from . import _trainer
from . import _util


class Local(object):
  """Class for local training, preprocessing and prediction."""

  def __init__(self, checkpoint=None):
    self._checkpoint = checkpoint
    if self._checkpoint is None:
      self._checkpoint = _util._DEFAULT_CHECKPOINT_GSURL

  def preprocess(self, input_csvs, labels_file, output_dir):
    """Local preprocessing with local DataFlow."""

    job_id = 'inception_preprocessed_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    p = beam.Pipeline('DirectPipelineRunner')
    _preprocess.configure_pipeline(p, self._checkpoint, input_csvs, labels_file,
                                   output_dir, job_id)
    p.run()

  def train(self, labels_file, input_dir, batch_size, max_steps, output_dir):
    """Local training."""

    num_classes = len(_util.get_labels(labels_file))
    model = _model.Model(num_classes, 0.5, self._checkpoint)
    task_data = {'type': 'master', 'index': 0}
    task = type('TaskSpec', (object,), task_data)
    _trainer.Trainer(input_dir, batch_size, max_steps, output_dir,
                     model, None, task).run_training()

  def predict(self, model_dir, image_files, labels_file):
    """Local prediction."""

    labels = _util.get_labels(labels_file)
    labels.append('UNKNOWN')
    model_dir = os.path.join(model_dir, 'model')
    with tf.Session() as sess:
      new_saver = tf.train.import_meta_graph(os.path.join(model_dir, 'export.meta'))
      new_saver.restore(sess, os.path.join(model_dir, 'export'))
      inputs = json.loads(tf.get_collection('inputs')[0])
      outputs = json.loads(tf.get_collection('outputs')[0])
      feed_dict = collections.defaultdict(list)
      for ii, image_filename in enumerate(image_files):
        with open(image_filename) as ff:
          image_bytes = ff.read()
          feed_dict[inputs['image_bytes']].append(image_bytes)
          feed_dict[inputs['key']].append(str(ii))
      predictions, scores = sess.run([outputs['prediction'], outputs['scores']],
                                     feed_dict=feed_dict)
    labels_and_scores = [(labels[predicted_index], class_scores[predicted_index])
                         for predicted_index, class_scores in zip(predictions, scores)]
    return labels_and_scores


  def batch_predict(self, model_dir, input_csv, labels_file, output_file, output_bq_table):
    """Local batch prediction."""

    labels = _util.get_labels(labels_file)
    labels.append('UNKNOWN')
    model_dir = os.path.join(model_dir, 'model')
    
    with tf.Session() as sess:
      new_saver = tf.train.import_meta_graph(os.path.join(model_dir, 'export.meta'))
      new_saver.restore(sess, os.path.join(model_dir, 'export'))
      inputs = json.loads(tf.get_collection('inputs')[0])
      outputs = json.loads(tf.get_collection('outputs')[0])
      feed_dict = collections.defaultdict(list)
      
      input_csv_f = ml.util._file.read_file_stream(input_csv)
      reader = csv.reader(input_csv_f)
      for ii, row in enumerate(reader):
        with ml.util._file.open_local_or_gcs(row[0], mode='r') as ff:
          image_bytes = ff.read()
          feed_dict[inputs['image_bytes']].append(image_bytes)
          feed_dict[inputs['key']].append(str(ii))
      predictions, scores = sess.run([outputs['prediction'], outputs['scores']],
                                     feed_dict=feed_dict)

    input_csv_f = ml.util._file.read_file_stream(input_csv)
    with ml.util._file.open_local_or_gcs(output_file, mode='w') as f_out:
      reader = csv.reader(input_csv_f)
      writer = csv.writer(f_out)
      for input, predicted_index, class_scores in zip(reader, predictions, scores):
        target_index = labels.index(input[1])
        target_prob = class_scores[target_index]
        predicted_prob = class_scores[predicted_index]
        writer.writerow(input + [labels[predicted_index]] + [str(target_prob)] +
                        [str(predicted_prob)] + map(str, class_scores))
    schema = [
        {'name': 'image_url', 'type': 'STRING'},
        {'name': 'target', 'type': 'STRING'},
        {'name': 'predicted', 'type': 'STRING'},
        {'name': 'target_prob', 'type': 'FLOAT'},
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
    return output_file, yaml_file
