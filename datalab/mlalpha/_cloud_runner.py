# Copyright 2016 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

import datetime
from googleapiclient import discovery

import datalab.context


# TODO(qimingj) Remove once the API is public since it will no longer be needed
_CLOUDML_DISCOVERY_URL = 'https://storage.googleapis.com/cloud-ml/discovery/' \
                         'ml_v1beta1_discovery.json'


class CloudRunner(object):
  """CloudML Trainer API Wrapper that takes a job_request, add authentication information,
     submit it to cloud, and get job response.
  """

  def __init__(self, job_request):
    """Initializes an instance of a LocalRunner

    Args:
      job_request: the arguments of the training job in a dict. For example,
          {
            'package_uris':  'gs://my-bucket/iris/trainer-0.1.tar.gz',
            'python_module': 'trainer.task',
            'scale_tier': 'BASIC',
            'region': 'us-central1',
            'args': {
              'train_data_paths': ['gs://mubucket/data/features_train'],
              'eval_data_paths': ['gs://mubucket/data/features_eval'],
              'metadata_path': 'gs://mubucket/data/metadata.yaml',
              'output_path': 'gs://mubucket/data/mymodel/',
            }
          }
    """

    self._job_request = dict(job_request)
    # convert job_args from dict to list as service required.
    if 'args' in job_request and isinstance(job_request['args'], dict):
      job_args = job_request['args']
      args = []
      for k,v in job_args.iteritems():
        if isinstance(v, list):
          for item in v:
            args.append('--' + k)
            args.append(str(item))
        else:
          args.append('--' + k)
          args.append(str(v))
      self._job_request['args'] = args

  def _create_default_job_name(self):
    job_name = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    if 'python_module' in self._job_request:
      job_name = self._job_request['python_module'].replace('.', '_') + \
          '_' + job_name
    return job_name

  def run(self, job_id=None):
    """Submit a training job to the CloudML service.

    Args:
      job_id: id for the training job. If None, a UUID will be generated.

    Returns: job info returned from service.
    """
    if job_id is None:
      job_id = self._create_default_job_name()
    job = {
        'job_id': job_id,
        'training_input': self._job_request,
    }
    context = datalab.context.Context.default()
    cloudml = discovery.build('ml', 'v1beta1', credentials=context.credentials,
                              discoveryServiceUrl=_CLOUDML_DISCOVERY_URL)
    request = cloudml.projects().jobs().create(body=job,
                                               parent='projects/' + context.project_id)
    request.headers['user-agent'] = 'GoogleCloudDataLab/1.0'
    return request.execute()
