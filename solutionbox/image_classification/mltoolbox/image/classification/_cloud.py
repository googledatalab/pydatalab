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


import base64
import datetime
import logging
import os
import urllib

from . import _util


_TF_GS_URL = 'gs://cloud-datalab/deploy/tf/tensorflow-1.0.0-cp27-cp27mu-manylinux1_x86_64.whl'
_PROTOBUF_GS_URL = 'gs://cloud-datalab/deploy/tf/protobuf-3.1.0-py2.py3-none-any.whl'


class Cloud(object):
  """Class for cloud training, preprocessing and prediction."""

  @staticmethod
  def preprocess(train_dataset, output_dir, eval_dataset, checkpoint, pipeline_option):
    """Preprocess data in Cloud with DataFlow."""

    import apache_beam as beam
    import google.datalab.utils
    from . import _preprocess

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
        'extra_packages': [staging_package_url, _TF_GS_URL, _PROTOBUF_GS_URL],
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'no_save_main_session': True
    }
    if pipeline_option is not None:
      options.update(pipeline_option)

    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    p = beam.Pipeline('DataflowRunner', options=opts)
    _preprocess.configure_pipeline(p, train_dataset, eval_dataset, checkpoint, output_dir, job_name)
    # suppress DataFlow warnings about wheel package as extra package.
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    original_level = logger.getEffectiveLevel()
    try:
      job_results = p.run()
    finally:
      logger.setLevel(original_level)
    if (_util.is_in_IPython()):
      import IPython
      dataflow_url = 'https://console.developers.google.com/dataflow?project=%s' % \
                     _util.default_project()
      html = 'Job "%s" submitted.' % job_name
      html += '<p>Click <a href="%s" target="_blank">here</a> to track preprocessing job. <br/>' \
          % dataflow_url
      IPython.display.display_html(html, raw=True)
    return google.datalab.utils.DataflowJob(job_results)

  @staticmethod
  def train(input_dir, batch_size, max_steps, output_dir, checkpoint, cloud_train_config):
    """Train model in the cloud with CloudML trainer service."""

    import google.datalab.ml as ml
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
      'package_uris': [staging_package_url, _TF_GS_URL, _PROTOBUF_GS_URL],
      'python_module': 'mltoolbox.image.classification.task',
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
  def predict(model_id, image_files, resize, show_image):
    """Predict using a deployed (online) model."""

    import google.datalab.ml as ml

    images = _util.load_images(image_files, resize=resize)

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
    return ret

  @staticmethod
  def batch_predict(dataset, model_dir, output_csv, output_bq_table, pipeline_option):
    """Batch predict running in cloud."""

    import apache_beam as beam
    import google.datalab.utils
    from . import _predictor

    if output_csv is None and output_bq_table is None:
      raise ValueError('output_csv and output_bq_table cannot both be None.')
    if 'temp_location' not in pipeline_option:
      raise ValueError('"temp_location" is not set in cloud.')

    job_name = ('batch-predict-image-classification-' +
                datetime.datetime.now().strftime('%y%m%d-%H%M%S'))
    staging_package_url = _util.repackage_to_staging(pipeline_option['temp_location'])
    options = {
        'staging_location': os.path.join(pipeline_option['temp_location'], 'staging'),
        'job_name': job_name,
        'project': _util.default_project(),
        'extra_packages': [staging_package_url, _TF_GS_URL, _PROTOBUF_GS_URL],
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'no_save_main_session': True
    }
    options.update(pipeline_option)

    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    p = beam.Pipeline('DataflowRunner', options=opts)
    _predictor.configure_pipeline(p, dataset, model_dir, output_csv, output_bq_table)
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    original_level = logger.getEffectiveLevel()
    try:
      job_results = p.run()
    finally:
      logger.setLevel(original_level)
    if (_util.is_in_IPython()):
      import IPython
      dataflow_url = ('https://console.developers.google.com/dataflow?project=%s' %
                      _util.default_project())
      html = 'Job "%s" submitted.' % job_name
      html += ('<p>Click <a href="%s" target="_blank">here</a> to track batch prediction job. <br/>'
               % dataflow_url)
      IPython.display.display_html(html, raw=True)
    return google.datalab.utils.DataflowJob(job_results)
