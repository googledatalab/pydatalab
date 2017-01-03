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
from googleapiclient import discovery
import google.cloud.ml as ml
import logging
import os

from . import _model
from . import _preprocess
from . import _trainer
from . import _util


_CLOUDML_DISCOVERY_URL = 'https://storage.googleapis.com/cloud-ml/discovery/' \
                         'ml_v1beta1_discovery.json'
_TF_GS_URL= 'gs://cloud-datalab/deploy/tf/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl'


class Cloud(object):
  """Class for cloud training, preprocessing and prediction."""

  def __init__(self, project, checkpoint=None):
    self._project = project
    self._checkpoint = checkpoint
    if self._checkpoint is None:
      self._checkpoint = _util._DEFAULT_CHECKPOINT_GSURL

  def preprocess(self, input_csvs, labels_file, output_dir, pipeline_option=None):
    """Cloud preprocessing with Cloud DataFlow."""

    job_name = 'preprocess-inception-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    options = {
        'staging_location': os.path.join(output_dir, 'tmp', 'staging'),
        'temp_location': os.path.join(output_dir, 'tmp'),
        'job_name': job_name,
        'project': self._project,
        'extra_packages': [ml.sdk_location, _util._PACKAGE_GS_URL, _TF_GS_URL],
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'no_save_main_session': True
    }
    if pipeline_option is not None:
      options.update(pipeline_option)

    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    p = beam.Pipeline('DataflowPipelineRunner', options=opts)
    _preprocess.configure_pipeline(
        p, self._checkpoint, input_csvs, labels_file, output_dir, job_name)
    p.run()

  def train(self, labels_file, input_dir, batch_size, max_steps, output_path, credentials,
            region, scale_tier):
    """Cloud training with CloudML trainer service."""

    num_classes = len(_util.get_labels(labels_file))
    job_id = 'inception_train_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    job_args_dict = {
      'input_dir': input_dir,
      'output_path': output_path,
      'max_steps': max_steps,
      'batch_size': batch_size,
      'num_classes': num_classes,
      'checkpoint': self._checkpoint
    }
    # convert job_args from dict to list as service required.
    job_args = []
    for k,v in job_args_dict.iteritems():
      if isinstance(v, list):
        for item in v:

          job_args.append('--' + k)
          job_args.append(str(item))
      else:
        job_args.append('--' + k)
        job_args.append(str(v))

    job_request = {
      'package_uris': _util._PACKAGE_GS_URL,
      'python_module': 'datalab_solutions.inception.task',
      'scale_tier': scale_tier,
      'region': region,
      'args': job_args
    }
    job = {
      'job_id': job_id,
      'training_input': job_request,
    }
    cloudml = discovery.build('ml', 'v1beta1', discoveryServiceUrl=_CLOUDML_DISCOVERY_URL,
        credentials=credentials)
    request = cloudml.projects().jobs().create(body=job,
                                               parent='projects/' + self._project)
    request.headers['user-agent'] = 'GoogleCloudDataLab/1.0'
    job_info = request.execute()
    return job_info

  def predict(self, model_id, image_files, labels_file, credentials):
    """Cloud prediction with CloudML prediction service."""

    labels = _util.get_labels(labels_file)
    data = []
    for ii, img_file in enumerate(image_files):
      with ml.util._file.open_local_or_gcs(img_file, 'rb') as f:
        img = base64.b64encode(f.read())
      data.append({
        'key': str(ii),
        'image_bytes': {'b64': img}
      })
    parts = model_id.split('.')
    if len(parts) != 2:
      raise Exception('Invalid model name for cloud prediction. Use "model.version".')    
    full_version_name = ('projects/%s/models/%s/versions/%s' % (self._project, parts[0], parts[1]))
    api = discovery.build('ml', 'v1beta1', credentials=credentials,
                          discoveryServiceUrl=_CLOUDML_DISCOVERY_URL)
    request = api.projects().predict(body={'instances': data}, name=full_version_name)
    job_results = request.execute()
    if 'predictions' not in job_results:
      raise Exception('Invalid response from service. Cannot find "predictions" in response.')
    predictions = job_results['predictions']
    labels_and_scores = [(labels[x['prediction']], x['scores'][x['prediction']])
                         for x in predictions]
    return labels_and_scores
