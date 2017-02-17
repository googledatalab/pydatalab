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


"""Provides interface for Datalab. It provides:
     local_preprocess
     local_train
     local_predict
     cloud_preprocess
     cloud_train
     cloud_predict
   Datalab will look for functions with the above names.
"""

import google.cloud.ml as ml
import logging
import os
import urllib

from . import _cloud
from . import _local
from . import _model
from . import _preprocess
from . import _trainer
from . import _util


def local_preprocess(train_dataset, output_dir, checkpoint=None, eval_dataset=None):
  """Preprocess data locally. Produce output that can be used by training efficiently.
  Args:
    train_dataset: training data source to preprocess. Can be CsvDataset or BigQueryDataSet.
        If eval_dataset is None, the pipeline will randomly split train_dataset into
        train/eval set with 7:3 ratio.
    output_dir: The output directory to use. Preprocessing will create a sub directory under
        it for each run, and also update "latest" file which points to the latest preprocessed
        directory. Users are responsible for cleanup. Can be local or GCS path.
    checkpoint: the Inception checkpoint to use.
    eval_dataset: evaluation data source to preprocess. Can be CsvDataset or BigQueryDataSet.
        If specified, it will be used for evaluation during training, and train_dataset will be
        completely used for training.
  """

  print 'Local preprocessing...'
  # TODO: Move this to a new process to avoid pickling issues
  # TODO: Expose train/eval split ratio
  _local.Local(checkpoint).preprocess(train_dataset, eval_dataset, output_dir)
  print 'Done'


def cloud_preprocess(train_dataset, output_dir, checkpoint=None, pipeline_option=None,
                     eval_dataset=None):
  """Preprocess data in Cloud with DataFlow.
     Produce output that can be used by training efficiently.
  Args:
    train_dataset: training data source to preprocess. Can be CsvDataset or BigQueryDataSet.
        For CsvDataSet, all files must be in GCS.
        If eval_dataset is None, the pipeline will randomly split train_dataset into
        train/eval set with 7:3 ratio.
    output_dir: The output directory to use. Preprocessing will create a sub directory under
        it for each run, and also update "latest" file which points to the latest preprocessed
        directory. Users are responsible for cleanup. GCS path only.
    checkpoint: the Inception checkpoint to use.
    pipeline_option: DataFlow pipeline options in a dictionary.
    eval_dataset: evaluation data source to preprocess. Can be CsvDataset or BigQueryDataSet.
        If specified, it will be used for evaluation during training, and train_dataset will be
        completely used for training.
  """

  job_name = _cloud.Cloud(checkpoint=checkpoint).preprocess(train_dataset, eval_dataset,
      output_dir, pipeline_option)
  if (_util.is_in_IPython()):
    import IPython
    
    dataflow_url = 'https://console.developers.google.com/dataflow?project=%s' % \
                   _util.default_project()
    html = 'Job "%s" submitted.' % job_name
    html += '<p>Click <a href="%s" target="_blank">here</a> to track preprocessing job. <br/>' \
        % dataflow_url
    IPython.display.display_html(html, raw=True)


def local_train(input_dir, batch_size, max_steps, output_dir, checkpoint=None):
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
  print 'Local training...'
  try:
    _local.Local(checkpoint).train(input_dir, batch_size, max_steps, output_dir)
  finally:
    logger.setLevel(original_level)
  print 'Done'


def cloud_train(input_dir, batch_size, max_steps, output_dir,
                cloud_train_config, checkpoint=None):
  """Train model in the cloud with CloudML trainer service.
     The output can be used for local prediction or for online deployment.
  Args:
    input_dir: A directory path containing preprocessed results. GCS path only.
    batch_size: size of batch used for training.
    max_steps: number of steps to train.
    output_dir: The output directory to use. GCS path only.
    cloud_train_config: a datalab.ml.CloudTrainingConfig object.
    checkpoint: the Inception checkpoint to use.
  """

  job = _cloud.Cloud(checkpoint=checkpoint).train(input_dir, batch_size,
      max_steps, output_dir, cloud_train_config)
  if (_util.is_in_IPython()):
    import IPython
    log_url_query_strings = {
      'project': _util.default_project(),
      'resource': 'ml.googleapis.com/job_id/' + job.info['jobId']
    }
    log_url = 'https://console.developers.google.com/logs/viewer?' + \
        urllib.urlencode(log_url_query_strings)
    html = 'Job "%s" submitted.' % job.info['jobId']
    html += '<p>Click <a href="%s" target="_blank">here</a> to view cloud log. <br/>' % log_url
    IPython.display.display_html(html, raw=True)


def _display_predict_results(results, show_image):
  if (_util.is_in_IPython()):
    import IPython
    for image_url, image, label_and_score in results:
      if show_image is True:
        IPython.display.display_html('<p style="font-size:28px">%s(%.5f)</p>' % label_and_score,
            raw=True)
        IPython.display.display(IPython.display.Image(data=image))
      else:
        IPython.display.display_html(
            '<p>%s&nbsp&nbsp&nbsp&nbsp%s(%.5f)</p>' % ((image_url,) + label_and_score), raw=True)
  else:
    print results


def local_predict(model_dir, image_files, resize=False, show_image=True):
  """Predict using an offline model.
  Args:
    model_dir: The directory of a trained inception model. Can be local or GCS paths.
    image_files: The paths to the image files to predict labels. Can be local or GCS paths.
    show_image: Whether to show images in the results.
    resize: Whether to resize the image to a reasonable size (300x300) before prediction.
  """
  print('Predicting...')
  images = _util.load_images(image_files, resize=resize)
  labels_and_scores = _local.Local().predict(model_dir, images)
  results = zip(image_files, images, labels_and_scores)
  _display_predict_results(results, show_image)
  print('Done')


def cloud_predict(model_id, image_files, resize=False, show_image=True):
  """Predict using a deployed (online) model.
  Args:
    model_id: The deployed model id in the form of "model.version".
    image_files: The paths to the image files to predict labels. GCS paths only.
    show_image: Whether to show images in the results.
    resize: Whether to resize the image to a reasonable size (300x300) before prediction.
        Set it to True if your images are too large to send over network.
  """
  print('Predicting...')
  images = _util.load_images(image_files, resize=resize)
  labels_and_scores = _cloud.Cloud().predict(model_id, images)
  results = zip(image_files, images, labels_and_scores)
  _display_predict_results(results, show_image)
  print('Done')


def local_batch_predict(dataset, model_dir, output_csv=None, output_bq_table=None):
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

  if output_csv is None and output_bq_table is None:
    raise ValueError('output_csv and output_bq_table cannot both be None.')

  print('Predicting...')
  _local.Local().batch_predict(dataset, model_dir, output_csv, output_bq_table)
  print('Done')


def cloud_batch_predict(dataset, model_dir, gcs_staging_location,
                        output_csv=None, output_bq_table=None, pipeline_option=None):
  """Batch predict running in cloud.

  Args:
    dataset: CsvDataSet or BigQueryDataSet for batch prediction input. Can contain either
        one column 'image_url', or two columns with another being 'label'.
    model_dir: A GCS path to a trained inception model directory.
    gcs_staging_location: A temporary location for DataFlow staging.
    output_csv: If specified, prediction results will be saved to the specified Csv file.
        It will also output a csv schema file with the name output_csv + '.schema.json'.
        GCS file path only.
    output_bq_table: If specified, prediction results will be saved to the specified BigQuery
        table. output_csv and output_bq_table can both be set, but cannot be both None.
    pipeline_option: DataFlow pipeline options in a dictionary.
  Raises:
    ValueError if both output_csv and output_bq_table are None.
  """

  if output_csv is None and output_bq_table is None:
    raise ValueError('output_csv and output_bq_table cannot both be None.')
  
  job_name = _cloud.Cloud().batch_predict(dataset, model_dir,
      gcs_staging_location, output_csv, output_bq_table, pipeline_option)
  if (_util.is_in_IPython()):
    import IPython
    
    dataflow_url = ('https://console.developers.google.com/dataflow?project=%s' %
                   _util.default_project())
    html = 'Job "%s" submitted.' % job_name
    html += ('<p>Click <a href="%s" target="_blank">here</a> to track batch prediction job. <br/>'
             % dataflow_url)
    IPython.display.display_html(html, raw=True)
