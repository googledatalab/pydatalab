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

  def preprocess(self, dataset, output_dir):
    """Local preprocessing with local DataFlow."""
    
    import datalab.mlalpha as mlalpha
    job_id = 'inception_preprocessed_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    options = {
        'project': _util.default_project(),
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    p = beam.Pipeline('DirectRunner', options=opts)
    if type(dataset) is mlalpha.CsvDataSet:
      _preprocess.configure_pipeline_csv(p, self._checkpoint, dataset.files, output_dir, job_id)
    elif type(dataset) is mlalpha.BigQueryDataSet:
      _preprocess.configure_pipeline_bigquery(p, self._checkpoint, dataset.sql, output_dir, job_id)
    else:
      raise ValueError('preprocess takes CsvDataSet or BigQueryDataset only.')
    p.run().wait_until_finish()

  def train(self, input_dir, batch_size, max_steps, output_dir):
    """Local training."""

    labels = _util.get_labels(input_dir)
    model = _model.Model(labels, 0.5, self._checkpoint)
    task_data = {'type': 'master', 'index': 0}
    task = type('TaskSpec', (object,), task_data)
    _trainer.Trainer(input_dir, batch_size, max_steps, output_dir,
                     model, None, task).run_training()

  def predict(self, model_dir, image_files):
    """Local prediction."""

    return _predictor.predict(model_dir, image_files)


  def batch_predict(self, model_dir, input_csv, output_file, output_bq_table):
    """Local batch prediction."""

    return _predictor.batch_predict(model_dir, input_csv, output_file, output_bq_table)
