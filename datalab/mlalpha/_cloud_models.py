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

"""Implements Cloud ML Model Operations"""

from googleapiclient import discovery
import os
import time
import yaml

import datalab.context
import datalab.storage
import datalab.utils


# TODO(qimingj) Remove once the API is public since it will no longer be needed
_CLOUDML_DISCOVERY_URL = 'https://storage.googleapis.com/cloud-ml/discovery/' \
                         'ml_v1beta1_discovery.json'


class CloudModels(object):
  """Represents a list of Cloud ML models for a project."""

  def __init__(self, project_id=None):
    """Initializes an instance of a CloudML Model list that is iteratable
           ("for model in CloudModels()").

    Args:
      project_id: project_id of the models. If not provided, default project_id will be used.
    """
    if project_id is None:
      project_id = datalab.context.Context.default().project_id
    self._project_id = project_id
    self._credentials = datalab.context.Context.default().credentials
    self._api = discovery.build('ml', 'v1alpha3', credentials=self._credentials,
                                discoveryServiceUrl=_CLOUDML_DISCOVERY_URL)

  def _retrieve_models(self, page_token, page_size):
    list_info = self._api.projects().models().list(
        parent='projects/' + self._project_id, pageToken=page_token, pageSize=page_size).execute()
    models = list_info.get('models', [])
    page_token = list_info.get('nextPageToken', None)
    return models, page_token

  def __iter__(self):
    return iter(datalab.utils.Iterator(self._retrieve_models))

  def get_model_details(self, model_name):
    """Get details of a model.

    Args:
      model_name: the name of the model. It can be a model full name
          ("projects/[project_id]/models/[model_name]") or just [model_name].
      Returns: a dictionary of the model details.
    """
    full_name = model_name
    if not model_name.startswith('projects/'):
      full_name = ('projects/%s/models/%s' % (self._project_id, model_name))
    return self._api.projects().models().get(name=full_name).execute()

  def create(self, model_name):
    """Create a model.

    Args:
      model_name: the short name of the model, such as "iris".
    """
    body = {'name': model_name}
    parent = 'projects/' + self._project_id
    self._api.projects().models().create(body=body, parent=parent).execute()

  def delete(self, model_name):
    """Delete a model.

    Args:
      model_name: the name of the model. It can be a model full name
          ("projects/[project_id]/models/[model_name]") or just [model_name].
    """
    full_name = model_name
    if not model_name.startswith('projects/'):
      full_name = ('projects/%s/models/%s' % (self._project_id, model_name))
    return self._api.projects().models().delete(name=full_name).execute()

  def list(self, count=10):
    """List models under the current project in a table view.

    Args:
      count: upper limit of the number of models to list.
    Raises:
      Exception if it is called in a non-IPython environment.
    """
    import IPython
    data = []
    # Add range(count) to loop so it will stop either it reaches count, or iteration
    # on self is exhausted. "self" is iterable (see __iter__() method).
    for _, model in zip(range(count), self):
      element = {'name': model['name']}
      if 'defaultVersion' in model:
        version_short_name = model['defaultVersion']['name'].split('/')[-1]
        element['defaultVersion'] = version_short_name
      data.append(element)

    IPython.display.display(
        datalab.utils.commands.render_dictionary(data, ['name', 'defaultVersion']))
    
  def describe(self, model_name):
    """Print information of a specified model.

    Args:
      model_name: the name of the model to print details on.
    """    
    model_yaml = yaml.safe_dump(self.get_model_details(model_name), default_flow_style=False)
    print model_yaml
    

class CloudModelVersions(object):
  """Represents a list of versions for a Cloud ML model."""

  def __init__(self, model_name, project_id=None):
    """Initializes an instance of a CloudML model version list that is iteratable
        ("for version in CloudModelVersions()").

    Args:
      model_name: the name of the model. It can be a model full name
          ("projects/[project_id]/models/[model_name]") or just [model_name].
      project_id: project_id of the models. If not provided and model_name is not a full name
          (not including project_id), default project_id will be used.
    """
    if project_id is None:
      self._project_id = datalab.context.Context.default().project_id
    self._credentials = datalab.context.Context.default().credentials
    self._api = discovery.build('ml', 'v1alpha3', credentials=self._credentials,
                                discoveryServiceUrl=_CLOUDML_DISCOVERY_URL)
    if not model_name.startswith('projects/'):
      model_name = ('projects/%s/models/%s' % (self._project_id, model_name))
    self._full_model_name = model_name
    self._model_name = self._full_model_name.split('/')[-1]

  def _retrieve_versions(self, page_token, page_size):
    parent = self._full_model_name
    list_info = self._api.projects().models().versions().list(parent=parent,
        pageToken=page_token, pageSize=page_size).execute()
    versions = list_info.get('versions', [])
    page_token = list_info.get('nextPageToken', None)
    return versions, page_token

  def __iter__(self):
    return iter(datalab.utils.Iterator(self._retrieve_versions))

  def get_version_details(self, version_name):
    """Get details of a version.

    Args:
      version: the name of the version in short form, such as "v1".
    Returns: a dictionary containing the version details.
    """
    name = ('%s/versions/%s' % (self._full_model_name, version_name))
    return self._api.projects().models().versions().get(name=name).execute()

  def _wait_for_long_running_operation(self, response):
    if 'name' not in response:
      raise Exception('Invaid response from service. Cannot find "name" field.')
    print('Waiting for job "%s"' % response['name'])
    while True:
      response = self._api.projects().operations().get(name=response['name']).execute()
      if 'done' not in response or response['done'] != True:
        time.sleep(3)
      else:
        if 'error' in response:
          print(response['error'])
        else:
          print('Done.')
        break

  def deploy(self, version_name, path):
    """Deploy a model version to the cloud.

    Args:
      version_name: the name of the version in short form, such as "v1".
      path: the Google Cloud Storage path (gs://...) which contains the model files.

    Raises: Exception if the path is invalid or does not contain expected files.
            Exception if the service returns invalid response.
    """
    if not path.startswith('gs://'):
      raise Exception('Invalid path. Only Google Cloud Storage path (gs://...) is accepted.')
    if not datalab.storage.Item.from_url(os.path.join(path, 'export.meta')).exists():
      # try appending '/model' sub dir.
      path = os.path.join(path, 'model')
      if not datalab.storage.Item.from_url(os.path.join(path, 'export.meta')).exists():
        raise Exception('Cannot find export.meta from given path.')

    body = {'name': self._model_name}
    parent = 'projects/' + self._project_id
    try:
      self._api.projects().models().create(body=body, parent=parent).execute()
    except:
      # Trying to create an already existing model gets an error. Ignore it.
      pass
    body = {
      'name': version_name,
      'deployment_uri': path,
    }
    response = self._api.projects().models().versions().create(body=body,
                   parent=self._full_model_name).execute()
    self._wait_for_long_running_operation(response)

  def delete(self, version_name):
    """Delete a version of model.

    Args:
      version_name: the name of the version in short form, such as "v1".
    """
    name = ('%s/versions/%s' % (self._full_model_name, version_name))
    response = self._api.projects().models().versions().delete(name=name).execute()
    self._wait_for_long_running_operation(response)

  def predict(self, version_name, data):
    """Get prediction results from features instances.

    Args:
      version_name: the name of the version used for prediction.
      data: typically a list of instance to be submitted for prediction. The format of the
          instance depends on the model. For example, structured data model may require
          a csv line for each instance.
    Returns:
      A list of prediction results for given instances. Each element is a dictionary representing
          output mapping from the graph.
      An example:
        [{"predictions": 1, "score": [0.00078, 0.71406, 0.28515]},
         {"predictions": 1, "score": [0.00244, 0.99634, 0.00121]}]
    """
    full_version_name = ('%s/versions/%s' % (self._full_model_name, version_name))
    request = self._api.projects().predict(body={'instances': data},
                                           name=full_version_name)
    request.headers['user-agent'] = 'GoogleCloudDataLab/1.0'
    result = request.execute()
    if 'predictions' not in result:
      raise Exception('Invalid response from service. Cannot find "predictions" in response.')

    return result['predictions']

  def describe(self, version_name):
    """Print information of a specified model.

    Args:
      version: the name of the version in short form, such as "v1".
    """
    version_yaml = yaml.safe_dump(self.get_version_details(version_name),
                                  default_flow_style=False)
    print version_yaml

  def list(self):
    """List versions under the current model in a table view.

    Raises:
      Exception if it is called in a non-IPython environment.
    """    
    import IPython

    # "self" is iterable (see __iter__() method).
    data = [{'name': version['name'].split()[-1], 
             'deploymentUri': version['deploymentUri'], 'createTime': version['createTime']}
            for version in self]
    IPython.display.display(
        datalab.utils.commands.render_dictionary(data, ['name', 'deploymentUri', 'createTime'])) 
