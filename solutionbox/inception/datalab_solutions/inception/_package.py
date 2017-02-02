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


def local_preprocess(dataset, output_dir, checkpoint=None):
  """Preprocess data locally. Produce output that can be used by training efficiently.
  Args:
    dataset: data source to preprocess. Can be either datalab.mlalpha.CsvDataset, or
        datalab.mlalpha.BigQueryDataSet.
    output_dir: The output directory to use. Preprocessing will create a sub directory under
        it for each run, and also update "latest" file which points to the latest preprocessed
        directory. Users are responsible for cleanup. Can be local or GCS path.
    checkpoint: the Inception checkpoint to use.
  """

  print 'Local preprocessing...'
  # TODO: Move this to a new process to avoid pickling issues
  # TODO: Expose train/eval split ratio
  _local.Local(checkpoint).preprocess(dataset, output_dir)
  print 'Done'


def cloud_preprocess(dataset, output_dir, checkpoint=None, pipeline_option=None):
  """Preprocess data in Cloud with DataFlow.
     Produce output that can be used by training efficiently.
  Args:
    dataset: data source to preprocess. Can be either datalab.mlalpha.CsvDataset, or
        datalab.mlalpha.BigQueryDataSet. For CsvDataSet, all files need to be in GCS.
    output_dir: The output directory to use. Preprocessing will create a sub directory under
        it for each run, and also update "latest" file which points to the latest preprocessed
        directory. Users are responsible for cleanup. GCS path only.
    checkpoint: the Inception checkpoint to use.
  """

  # TODO: Move this to a new process to avoid pickling issues
  # TODO: Expose train/eval split ratio
  job_name = _cloud.Cloud(checkpoint=checkpoint).preprocess(dataset, output_dir, pipeline_option)
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
                region, scale_tier='BASIC', checkpoint=None):
  """Train model in the cloud with CloudML trainer service.
     The output can be used for local prediction or for online deployment.
  Args:
    input_dir: A directory path containing preprocessed results. GCS path only.
    batch_size: size of batch used for training.
    max_steps: number of steps to train.
    output_dir: The output directory to use. GCS path only.
    checkpoint: the Inception checkpoint to use.
  """

  job_info = _cloud.Cloud(checkpoint=checkpoint).train(input_dir, batch_size,
      max_steps, output_dir, region, scale_tier)
  if (_util.is_in_IPython()):
    import IPython
    log_url_query_strings = {
      'project': _util.default_project(),
      'resource': 'ml.googleapis.com/job_id/' + job_info['jobId']
    }
    log_url = 'https://console.developers.google.com/logs/viewer?' + \
        urllib.urlencode(log_url_query_strings)
    html = 'Job "%s" submitted.' % job_info['jobId']
    html += '<p>Click <a href="%s" target="_blank">here</a> to view cloud log. <br/>' % log_url
    IPython.display.display_html(html, raw=True)


def _display_predict_results(results, show_image):
  if (_util.is_in_IPython()):
    import IPython
    for image_file, label_and_score in results:
      if show_image is True:
        IPython.display.display_html('<p style="font-size:28px">%s(%.5f)</p>' % label_and_score,
            raw=True)
        with ml.util._file.open_local_or_gcs(image_file, mode='r') as f:
          IPython.display.display(IPython.display.Image(data=f.read()))
      else:
        IPython.display.display_html(
            '<p>%s&nbsp&nbsp%s(%.5f)</p>' % ((image_file,) + label_and_score), raw=True)
  else:
    print results


def local_predict(model_dir, image_files, show_image=True):
  """Predict using an offline model.
  Args:
    model_dir: The directory of a trained inception model. Can be local or GCS paths.
    image_files: The paths to the image files to predict labels. Can be local or GCS paths.
    show_image: Whether to show images in the results.
  """
  print('Predicting...')
  labels_and_scores = _local.Local().predict(model_dir, image_files)
  results = zip(image_files, labels_and_scores)
  _display_predict_results(results, show_image)
  print('Done')


def cloud_predict(model_id, image_files, show_image=True):
  """Predict using a deployed (online) model.
  Args:
    model_id: The deployed model id in the form of "model.version".
    image_files: The paths to the image files to predict labels. GCS paths only.
    show_image: Whether to show images in the results.
  """
  print('Predicting...')
  labels_and_scores = _cloud.Cloud().predict(model_id, image_files)
  results = zip(image_files, labels_and_scores)
  _display_predict_results(results, show_image)


def local_batch_predict(model_dir, input_csv, output_file, output_bq_table=None):
  """Batch predict using an offline model.
  Args:
    model_dir: The directory of a trained inception model. Can be local or GCS paths.
    input_csv: The input csv which include two columns only: image_gs_url, label.
        Can be local or GCS paths.
    output_file: The output csv file containing prediction results.
    output_bq_table: If provided, will also save the results to BigQuery table.
  """
  print('Predicting...')
  _local.Local().batch_predict(model_dir, input_csv, output_file, output_bq_table)
  print('Done')

def cloud_batch_predict(model_dir, image_files, show_image=True, output_file=None):
  """Not Implemented Yet"""
  pass
