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
import datetime
import json
import os
import tensorflow as tf

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

  def train(self, labels_file, input_dir, batch_size, max_steps, output_path):
    """Local training."""

    num_classes = len(_util.get_labels(labels_file))
    model = _model.Model(num_classes, 0.5, self._checkpoint)
    task_data = {'type': 'master', 'index': 0}
    task = type('TaskSpec', (object,), task_data)
    _trainer.Trainer(input_dir, batch_size, max_steps, output_path,
                     model, None, task).run_training()

  def predict(self, model_dir, image_files, labels_file):
    """Local prediction."""
    labels = _util.get_labels(labels_file)
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
