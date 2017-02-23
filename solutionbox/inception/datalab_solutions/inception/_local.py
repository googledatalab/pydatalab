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
import json
import os
import tensorflow as tf
import yaml


from . import _model
from . import _predictor
from . import _preprocess
from . import _trainer
from . import _util


class Local(object):
  """Class for local training, preprocessing and prediction."""

  def __init__(self, checkpoint=None):
    self._checkpoint = checkpoint
    if self._checkpoint is None:
      self._checkpoint = _util._DEFAULT_CHECKPOINT_GSURL

  def preprocess(self, train_dataset, eval_dataset, output_dir):
    """Local preprocessing with local DataFlow."""
    
    import datalab.ml as ml
    job_id = 'inception_preprocessed_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    # Project is needed for bigquery data source, even in local run.
    options = {
        'project': _util.default_project(),
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    p = beam.Pipeline('DirectRunner', options=opts)
    _preprocess.configure_pipeline(p, train_dataset, eval_dataset,
        self._checkpoint, output_dir, job_id)
    p.run().wait_until_finish()

  def train(self, input_dir, batch_size, max_steps, output_dir):
    """Local training."""

    labels = _util.get_labels(input_dir)
    model = _model.Model(labels, 0.5, self._checkpoint)
    task_data = {'type': 'master', 'index': 0}
    task = type('TaskSpec', (object,), task_data)
    _trainer.Trainer(input_dir, batch_size, max_steps, output_dir,
                     model, None, task).run_training()

  def predict(self, model_dir, images):
    """Local prediction."""

    return _predictor.predict(model_dir, images)


  def batch_predict(self, dataset, model_dir, output_csv, output_bq_table):
    """Local batch prediction."""
    import datalab.ml as ml
    job_id = 'inception_batch_predict_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    # Project is needed for bigquery data source, even in local run.
    options = {
        'project': _util.default_project(),
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    p = beam.Pipeline('DirectRunner', options=opts)
    _predictor.configure_pipeline(p, dataset, model_dir, output_csv, output_bq_table)
    p.run().wait_until_finish()
