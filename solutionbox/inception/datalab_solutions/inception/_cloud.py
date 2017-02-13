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


_TF_GS_URL= 'gs://cloud-datalab/deploy/tf/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl'


class Cloud(object):
  """Class for cloud training, preprocessing and prediction."""

  def __init__(self, checkpoint=None):
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
        'project': _util.default_project(),
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

  def train(self, labels_file, input_dir, batch_size, max_steps, output_path,
            region, scale_tier):
    """Cloud training with CloudML trainer service."""

    import datalab.mlalpha as mlalpha
    num_classes = len(_util.get_labels(labels_file))
    job_args = {
      'input_dir': input_dir,
      'output_path': output_path,
      'max_steps': max_steps,
      'batch_size': batch_size,
      'num_classes': num_classes,
      'checkpoint': self._checkpoint
    }
    job_request = {
      'package_uris': _util._PACKAGE_GS_URL,
      'python_module': 'datalab_solutions.inception.task',
      'scale_tier': scale_tier,
      'region': region,
      'args': job_args
    }
    cloud_runner = mlalpha.CloudRunner(job_request)
    job_id = 'inception_train_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    return cloud_runner.run(job_id)

  def predict(self, model_id, image_files, labels_file):
    """Cloud prediction with CloudML prediction service."""

    import datalab.mlalpha as mlalpha
    parts = model_id.split('.')
    if len(parts) != 2:
      raise Exception('Invalid model name for cloud prediction. Use "model.version".')

    labels = _util.get_labels(labels_file)
    labels.append('UNKNOWN')
    data = []
    for ii, img_file in enumerate(image_files):
      with ml.util._file.open_local_or_gcs(img_file, 'rb') as f:
        img = base64.b64encode(f.read())
      data.append({
        'key': str(ii),
        'image_bytes': {'b64': img}
      })

    cloud_predictor = mlalpha.CloudPredictor(parts[0], parts[1])
    predictions = cloud_predictor.predict(data)
    labels_and_scores = [(labels[x['prediction']], x['scores'][x['prediction']])
                         for x in predictions]
    return labels_and_scores
