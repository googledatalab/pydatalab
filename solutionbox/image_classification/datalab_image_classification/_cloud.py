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


"""Cloud implementation for preprocessing, training and prediction for inception model.
"""

import apache_beam as beam
import base64
import collections
import datetime
import logging
import os
import urllib

from . import _model
from . import _predictor
from . import _preprocess
from . import _trainer
from . import _util


class Cloud(object):
  """Class for cloud training, preprocessing and prediction."""

  @staticmethod
  def preprocess(train_dataset, output_dir, eval_dataset=None, checkpoint=None,
                 pipeline_option=None):
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
      eval_dataset: evaluation data source to preprocess. Can be CsvDataset or BigQueryDataSet.
          If specified, it will be used for evaluation during training, and train_dataset will be
          completely used for training.
      checkpoint: the Inception checkpoint to use.
      pipeline_option: DataFlow pipeline options in a dictionary.
    Returns:
      the job name of the DataFlow job.
    """

    if checkpoint is None:
      checkpoint = _util._DEFAULT_CHECKPOINT_GSURL

    job_name = ('preprocess-image-classification-' +
                datetime.datetime.now().strftime('%y%m%d-%H%M%S'))
    staging_package_url = _util.repackage_to_staging(output_dir)
    options = {
        'staging_location': os.path.join(output_dir, 'tmp', 'staging'),
        'temp_location': os.path.join(output_dir, 'tmp'),
        'job_name': job_name,
        'project': _util.default_project(),
        'extra_packages': [staging_package_url],
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'no_save_main_session': True
    }
    if pipeline_option is not None:
      options.update(pipeline_option)

    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    p = beam.Pipeline('DataflowRunner', options=opts)
    _preprocess.configure_pipeline(p, train_dataset, eval_dataset, checkpoint,
        output_dir, job_name)
    p.run()
    if (_util.is_in_IPython()):
      import IPython
      dataflow_url = 'https://console.developers.google.com/dataflow?project=%s' % \
                     _util.default_project()
      html = 'Job "%s" submitted.' % job_name
      html += '<p>Click <a href="%s" target="_blank">here</a> to track preprocessing job. <br/>' \
          % dataflow_url
      IPython.display.display_html(html, raw=True)
    return job_name

  @staticmethod
  def train(input_dir, batch_size, max_steps, output_dir, cloud_train_config, checkpoint=None):
    """Train model in the cloud with CloudML trainer service.
       The output can be used for local prediction or for online deployment.
    Args:
      input_dir: A directory path containing preprocessed results. GCS path only.
      batch_size: size of batch used for training.
      max_steps: number of steps to train.
      output_dir: The output directory to use. GCS path only.
      cloud_train_config: a datalab.ml.CloudTrainingConfig object.
      checkpoint: the Inception checkpoint to use.
    Returns:
      the job name of the training job.
    """

    import datalab.ml as ml
    if checkpoint is None:
      checkpoint = _util._DEFAULT_CHECKPOINT_GSURL    
    staging_package_url = _util.repackage_to_staging(output_dir)
    job_args = {
      'input_dir': input_dir,
      'max_steps': max_steps,
      'batch_size': batch_size,
      'checkpoint': checkpoint
    }
    job_request = {
      'package_uris': [staging_package_url],
      'python_module': 'datalab_image_classification.task',
      'job_dir': output_dir,
      'args': job_args
    }
    job_request.update(dict(cloud_train_config._asdict()))
    job_id = 'image_classification_train_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    job = ml.Job.submit_training(job_request, job_id)
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
    return job

  @staticmethod
  def predict(model_id, image_files, resize=False, show_image=True):
    """Predict using a deployed (online) model.
    Args:
      model_id: The deployed model id in the form of "model.version".
      image_files: The paths to the image files to predict labels. GCS paths only.
      show_image: Whether to show images in the results.
      resize: Whether to resize the image to a reasonable size (300x300) before prediction.
          Set it to True if your images are too large to send over network.
    """

    import datalab.ml as ml

    images = _util.load_images(image_files, resize=resize)

    print('Predicting...')
    parts = model_id.split('.')
    if len(parts) != 2:
      raise ValueError('Invalid model name for cloud prediction. Use "model.version".')
    if len(images) == 0:
      raise ValueError('images is empty.')

    data = []
    for ii, image in enumerate(images):
      image_encoded = base64.b64encode(image)
      data.append({
        'key': str(ii),
        'image_bytes': {'b64': image_encoded}
      })

    predictions = ml.ModelVersions(parts[0]).predict(parts[1], data)
    if len(predictions) == 0:
      raise Exception('Prediction results are empty.')
    # Although prediction results contains a labels list in each instance, they are all the same
    # so taking the first one.
    labels = predictions[0]['labels']
    labels_and_scores = [(x['prediction'], x['scores'][labels.index(x['prediction'])])
                         for x in predictions]
    results = zip(image_files, images, labels_and_scores)
    ret = _util.process_prediction_results(results, show_image)
    print('Done')
    return ret

  @staticmethod
  def batch_predict(dataset, model_dir, gcs_staging_location,
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
    Returns:
      the job name of the DataFlow job.
    Raises:
      ValueError if both output_csv and output_bq_table are None.
    """

    if output_csv is None and output_bq_table is None:
      raise ValueError('output_csv and output_bq_table cannot both be None.')

    job_name = ('batch-predict-image-classification-' +
                datetime.datetime.now().strftime('%y%m%d-%H%M%S'))
    staging_package_url = _util.repackage_to_staging(gcs_staging_location)
    options = {
        'staging_location': os.path.join(gcs_staging_location, 'tmp', 'staging'),
        'temp_location': os.path.join(gcs_staging_location, 'tmp'),
        'job_name': job_name,
        'project': _util.default_project(),
        'extra_packages': [staging_package_url],
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'no_save_main_session': True
    }
    if pipeline_option is not None:
      options.update(pipeline_option)

    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    p = beam.Pipeline('DataflowRunner', options=opts)
    _predictor.configure_pipeline(p, dataset, model_dir, output_csv, output_bq_table)
    p.run()
    if (_util.is_in_IPython()):
      import IPython
      dataflow_url = ('https://console.developers.google.com/dataflow?project=%s' %
                     _util.default_project())
      html = 'Job "%s" submitted.' % job_name
      html += ('<p>Click <a href="%s" target="_blank">here</a> to track batch prediction job. <br/>'
               % dataflow_url)
      IPython.display.display_html(html, raw=True)
    return job_name
