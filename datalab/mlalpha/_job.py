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

import datalab.utils
import datalab.context
from googleapiclient import discovery


# TODO(qimingj) Remove once the API is public since it will no longer be needed
_CLOUDML_DISCOVERY_URL = 'https://storage.googleapis.com/cloud-ml/discovery/' \
                         'ml_v1beta1_discovery.json'


class Job(object):
  """Represents a Cloud ML job."""

  def __init__(self, name, context=None, api=None):
    """Initializes an instance of a CloudML Job.

    Args:
      name: the name of the job. It can be an operation full name
          ("projects/[project_id]/operations/[operation_name]") or just [operation_name].
      context: an optional Context object providing project_id and credentials.
      api: optional CloudML API client.
    """
    if context is None:
      context = datalab.context.Context.default()
    self._context = context
    if api is None:
      api = discovery.build('ml', 'v1beta1', credentials=self._context.credentials,
                            discoveryServiceUrl=_CLOUDML_DISCOVERY_URL)
    self._api = api
    if not name.startswith('projects/'):
      name = 'projects/' + self._context.project_id + '/jobs/' + name
    self._name = name
    self.refresh()

  @property
  def info(self):
    return self._info

  def refresh(self):
    """ Refresh the job info. """
    self._info = self._api.projects().jobs().get(name=self._name).execute()


class Jobs(object):
  """Represents a list of Cloud ML jobs for a project."""

  def __init__(self, filter=None, context=None, api=None):
    """Initializes an instance of a CloudML Job list that is iteratable ("for job in jobs()").

    Args:
      filter: filter string for retrieving jobs. Currently only "done=true|false" is supported.
      context: an optional Context object providing project_id and credentials.
      api: an optional CloudML API client.
    """
    self._filter = filter
    if context is None:
      context = datalab.context.Context.default()
    self._context = context
    if api is None:
      api = discovery.build('ml', 'v1beta1', credentials=self._context.credentials,
                            discoveryServiceUrl=_CLOUDML_DISCOVERY_URL)
    self._api = api

  def _retrieve_jobs(self, page_token, page_size):
    list_info = self._api.projects().jobs().list(parent='projects/' + self._context.project_id,
                                                 pageToken=page_token, pageSize=page_size,
                                                 filter=self._filter).execute()
    jobs = list_info.get('jobs', [])
    page_token = list_info.get('nextPageToken', None)
    return jobs, page_token

  def __iter__(self):
    return iter(datalab.utils.Iterator(self._retrieve_jobs))

  def get_job_by_name(self, name):
    """ get a CloudML job by its name.
    Args:
      name: the name of the job. See "Job" class constructor.
    """
    return Job(name, self._context, self._api)

