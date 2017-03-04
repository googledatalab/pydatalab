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

"""Implements Cloud ML Operation wrapper."""


import google.datalab as datalab
from googleapiclient import discovery
import yaml


class Job(datalab.Job):
  """Represents a Cloud ML job."""

  def __init__(self, name, context=None):
    """Initializes an instance of a CloudML Job.

    Args:
      name: the name of the job. It can be an operation full name
          ("projects/[project_id]/jobs/[operation_name]") or just [operation_name].
      context: an optional Context object providing project_id and credentials.
    """
    super(Job, self).__init__(name)
    if context is None:
      context = datalab.Context.default()
    self._context = context
    self._api = discovery.build('ml', 'v1', credentials=self._context.credentials)
    if not name.startswith('projects/'):
      name = 'projects/' + self._context.project_id + '/jobs/' + name
    self._name = name
    self._refresh_state()

  def _refresh_state(self):
    """ Refresh the job info. """
    self._info = self._api.projects().jobs().get(name=self._name).execute()
    self._fatal_error = self._info.get('errorMessage', None)
    state = str(self._info.get('state'))
    self._is_complete = (state == 'SUCCEEDED' or state == 'FAILED')

  @property
  def info(self):
    self._refresh_state()
    return self._info

  def describe(self):
    self._refresh_state()
    job_yaml = yaml.safe_dump(self._info, default_flow_style=False)
    print job_yaml

  @staticmethod 
  def submit_training(job_request, job_id=None):
    """Submit a training job.

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
          If 'args' is present in job_request and is a dict, it will be expanded to
          --key value or --key list_item_0 --key list_item_1, ...
      job_id: id for the training job. If None, an id based on timestamp will be generated.
    Returns:
      A Job object representing the cloud training job.
    """
    new_job_request = dict(job_request)
    # convert job_args from dict to list as service required.
    if 'args' in job_request and isinstance(job_request['args'], dict):
      job_args = job_request['args']
      args = []
      for k,v in job_args.iteritems():
        if isinstance(v, list):
          for item in v:
            args.append('--' + str(k))
            args.append(str(item))
        else:
          args.append('--' + str(k))
          args.append(str(v))
      new_job_request['args'] = args
    
    if job_id is None:
      job_id = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
      if 'python_module' in new_job_request:
        job_id = new_job_request['python_module'].replace('.', '_') + \
            '_' + job_id
        
    job = {
        'job_id': job_id,
        'training_input': new_job_request,
    }
    context = datalab.Context.default()
    cloudml = discovery.build('ml', 'v1', credentials=context.credentials)
    request = cloudml.projects().jobs().create(body=job,
                                               parent='projects/' + context.project_id)
    request.headers['user-agent'] = 'GoogleCloudDataLab/1.0'
    request.execute()
    return Job(job_id)


class Jobs(object):
  """Represents a list of Cloud ML jobs for a project."""

  def __init__(self, filter=None):
    """Initializes an instance of a CloudML Job list that is iteratable ("for job in jobs()").

    Args:
      filter: filter string for retrieving jobs, such as "state=FAILED"
      context: an optional Context object providing project_id and credentials.
      api: an optional CloudML API client.
    """
    self._filter = filter
    self._context = datalab.Context.default()
    self._api = discovery.build('ml', 'v1', credentials=self._context.credentials)

  def _retrieve_jobs(self, page_token, page_size):
    list_info = self._api.projects().jobs().list(parent='projects/' + self._context.project_id,
                                                 pageToken=page_token, pageSize=page_size,
                                                 filter=self._filter).execute()
    jobs = list_info.get('jobs', [])
    page_token = list_info.get('nextPageToken', None)
    return jobs, page_token

  def get_iterator(self):
    """Get iterator of jobs so it can be used as "for model in Jobs().get_iterator()".
    """
    return iter(datalab.utils.Iterator(self._retrieve_jobs))

  def list(self, count=10):
    import IPython
    data = [{'Id': job['jobId'], 'State': job.get('state', 'UNKNOWN'),
             'createTime': job['createTime']}
            for _, job in zip(range(count), self)]
    IPython.display.display(
        datalab.utils.commands.render_dictionary(data, ['Id', 'State', 'createTime']))
