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
import logging
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

  @staticmethod
  def preprocess(train_dataset, output_dir, eval_dataset=None, checkpoint=None):
    """Preprocess data locally. Produce output that can be used by training efficiently.
    Args:
      train_dataset: training data source to preprocess. Can be CsvDataset or BigQueryDataSet.
          If eval_dataset is None, the pipeline will randomly split train_dataset into
          train/eval set with 7:3 ratio.
      output_dir: The output directory to use. Preprocessing will create a sub directory under
          it for each run, and also update "latest" file which points to the latest preprocessed
          directory. Users are responsible for cleanup. Can be local or GCS path.
      eval_dataset: evaluation data source to preprocess. Can be CsvDataset or BigQueryDataSet.
          If specified, it will be used for evaluation during training, and train_dataset will be
          completely used for training.
      checkpoint: the Inception checkpoint to use.
  """
    print('Local preprocessing...')
    if checkpoint is None:
      checkpoint = _util._DEFAULT_CHECKPOINT_GSURL
    job_id = 'inception_preprocessed_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    # Project is needed for bigquery data source, even in local run.
    options = {
        'project': _util.default_project(),
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    p = beam.Pipeline('DirectRunner', options=opts)
    _preprocess.configure_pipeline(p, train_dataset, eval_dataset,
        checkpoint, output_dir, job_id)
    p.run().wait_until_finish()
    print('Done')

  @staticmethod
  def train(input_dir, batch_size, max_steps, output_dir, checkpoint=None):
    """Train model locally. The output can be used for local prediction or for online deployment.
    Args:
      input_dir: A directory path containing preprocessed results. Can be local or GCS path.
      batch_size: size of batch used for training.
      max_steps: number of steps to train.
      output_dir: The output directory to use. Can be local or GCS path.
      checkpoint: the Inception checkpoint to use.
    """

    logger = logging.getLogger()
    original_level = logger.getEffectiveLevel()
    logger.setLevel(logging.INFO)
    print('Local training...')
    if checkpoint is None:
      checkpoint = _util._DEFAULT_CHECKPOINT_GSURL
    try:
      labels = _util.get_labels(input_dir)
      model = _model.Model(labels, 0.5, checkpoint)
      task_data = {'type': 'master', 'index': 0}
      task = type('TaskSpec', (object,), task_data)
      _trainer.Trainer(input_dir, batch_size, max_steps, output_dir,
                       model, None, task).run_training()
    finally:
      logger.setLevel(original_level)
    print('Done')

  @staticmethod
  def predict(model_dir, image_files, resize=False, show_image=True):
    """Predict using an model in a local or GCS directory.
    Args:
      model_dir: The directory of a trained inception model. Can be local or GCS paths.
      image_files: The paths to the image files to predict labels. Can be local or GCS paths.
      resize: Whether to resize the image to a reasonable size (300x300) before prediction.
      show_image: Whether to show images in the results.
    """

    print('Predicting...')
    images = _util.load_images(image_files, resize=resize)
    labels_and_scores = _predictor.predict(model_dir, images)
    results = zip(image_files, images, labels_and_scores)
    ret = _util.process_prediction_results(results, show_image)
    print('Done')
    return ret

  @staticmethod
  def batch_predict(dataset, model_dir, output_csv=None, output_bq_table=None):
    """Batch predict running locally.
    Args:
      dataset: CsvDataSet or BigQueryDataSet for batch prediction input. Can contain either
          one column 'image_url', or two columns with another being 'label'.
      model_dir: The directory of a trained inception model. Can be local or GCS paths.
      output_csv: The output csv file for prediction results. If specified,
          it will also output a csv schema file with the name output_csv + '.schema.json'.
      output_bq_table: if specified, the output BigQuery table for prediction results.
          output_csv and output_bq_table can both be set.
    Raises:
      ValueError if both output_csv and output_bq_table are None.
    """

    print('Predicting...')
    if output_csv is None and output_bq_table is None:
      raise ValueError('output_csv and output_bq_table cannot both be None.')

    job_id = 'inception_batch_predict_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')

    # Project is needed for bigquery data source, even in local run.
    options = {
        'project': _util.default_project(),
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    p = beam.Pipeline('DirectRunner', options=opts)
    _predictor.configure_pipeline(p, dataset, model_dir, output_csv, output_bq_table)
    p.run().wait_until_finish()
    print('Done')
