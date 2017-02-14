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
import google.cloud.ml as ml
import logging
import os


from . import _model
from . import _predictor
from . import _preprocess
from . import _trainer
from . import _util


_TF_GS_URL= 'gs://cloud-datalab/deploy/tf/tensorflow-0.12.0rc1-cp27-none-linux_x86_64.whl'
# Keep in sync with "data_files" in package's setup.py
_SETUP_PY = '/datalab/packages_setup/inception/setup.py'

class Cloud(object):
  """Class for cloud training, preprocessing and prediction."""

  def __init__(self, checkpoint=None):
    self._checkpoint = checkpoint
    if self._checkpoint is None:
      self._checkpoint = _util._DEFAULT_CHECKPOINT_GSURL

  def _repackage_to_staging(self, output_path):
    """Repackage inception from local installed location and copy it to GCS.
    """

    import datalab.mlalpha as mlalpha

    # Find the package root. __file__ is under [package_root]/datalab_solutions/inception.
    package_root = os.path.join(os.path.dirname(__file__), '../../')
    staging_package_url = os.path.join(output_path, 'staging', 'inception.tar.gz')
    mlalpha.package_and_copy(package_root, _SETUP_PY, staging_package_url)
    return staging_package_url

  def preprocess(self, train_dataset, eval_dataset, output_dir, pipeline_option):
    """Cloud preprocessing with Cloud DataFlow."""

    import datalab.mlalpha as mlalpha

    job_name = 'preprocess-inception-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    staging_package_url = self._repackage_to_staging(output_dir)
    options = {
        'staging_location': os.path.join(output_dir, 'tmp', 'staging'),
        'temp_location': os.path.join(output_dir, 'tmp'),
        'job_name': job_name,
        'project': _util.default_project(),
        'extra_packages': [ml.sdk_location, staging_package_url, _TF_GS_URL],
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'no_save_main_session': True
    }
    if pipeline_option is not None:
      options.update(pipeline_option)

    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    p = beam.Pipeline('DataflowRunner', options=opts)
    _preprocess.configure_pipeline(p, train_dataset, eval_dataset, self._checkpoint,
        output_dir, job_name)
    p.run()
    return job_name

  def train(self, input_dir, batch_size, max_steps, output_path, cloud_train_config):
    """Cloud training with CloudML trainer service."""

    import datalab.mlalpha as mlalpha
    
    staging_package_url = self._repackage_to_staging(output_path)
    job_args = {
      'input_dir': input_dir,
      'output_path': output_path,
      'max_steps': max_steps,
      'batch_size': batch_size,
      'checkpoint': self._checkpoint
    }
    job_request = {
      'package_uris': staging_package_url,
      'python_module': 'datalab_solutions.inception.task',
      'args': job_args
    }
    job_request.update(dict(cloud_train_config._asdict()))
    job_id = 'inception_train_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    job = mlalpha.Job.submit_training(job_request, job_id)
    return job

  def predict(self, model_id, image_files):
    """Cloud prediction with CloudML prediction service."""

    import datalab.mlalpha as mlalpha
    parts = model_id.split('.')
    if len(parts) != 2:
      raise Exception('Invalid model name for cloud prediction. Use "model.version".')

    data = []
    for ii, img_file in enumerate(image_files):
      with ml.util._file.open_local_or_gcs(img_file, 'rb') as f:
        img = base64.b64encode(f.read())
      data.append({
        'key': str(ii),
        'image_bytes': {'b64': img}
      })

    predictions = mlalpha.CloudModelVersions(parts[0]).predict(parts[1], data)
    if len(predictions) == 0:
      raise Exception('Prediction results are empty.')
    # Although prediction results contains a labels list in each instance, they are all the same
    # so taking the first one.
    labels = predictions[0]['labels']
    labels_and_scores = [(x['prediction'], x['scores'][labels.index(x['prediction'])])
                         for x in predictions]
    return labels_and_scores

  def batch_predict(self, dataset, model_dir, gcs_staging_location, output_csv,
                    output_bq_table, pipeline_option):
    """Cloud batch prediction with a model specified by a GCS directory."""

    import datalab.mlalpha as mlalpha

    job_name = 'batch-predict-inception-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    staging_package_url = self._repackage_to_staging(gcs_staging_location)
    options = {
        'staging_location': os.path.join(gcs_staging_location, 'tmp', 'staging'),
        'temp_location': os.path.join(gcs_staging_location, 'tmp'),
        'job_name': job_name,
        'project': _util.default_project(),
        'extra_packages': [ml.sdk_location, staging_package_url, _TF_GS_URL],
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'no_save_main_session': True
    }
    if pipeline_option is not None:
      options.update(pipeline_option)

    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    p = beam.Pipeline('DataflowRunner', options=opts)
    _predictor.configure_pipeline(p, dataset, model_dir, output_csv, output_bq_table)
    p.run()
    return job_name
