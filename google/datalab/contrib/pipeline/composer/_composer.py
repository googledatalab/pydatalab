# Copyright 2017 Google Inc. All rights reserved.
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

import google.cloud.storage as gcs
import re
from google.datalab.contrib.pipeline.composer._api import Api


class Composer(object):
  """ Represents a Composer object that encapsulates a set of functionality relating to the
  Cloud Composer service.

  This object can be used to generate the python airflow spec.
  """

  gcs_file_regexp = re.compile('gs://.*')

  def __init__(self, zone, environment):
    """ Initializes an instance of a Composer object.

    Args:
      zone: Zone in which Composer environment has been created.
      environment: Name of the Composer environment.
    """
    self._zone = zone
    self._environment = environment
    self._gcs_dag_location = None

  def deploy(self, name, dag_string):
    client = gcs.Client()
    _, _, bucket_name, file_path = self.gcs_dag_location.split('/', 3)  # setting maxsplit to 3
    file_name = '{0}{1}.py'.format(file_path, name)

    bucket = client.get_bucket(bucket_name)
    blob = gcs.Blob(file_name, bucket)
    blob.upload_from_string(dag_string)

  @property
  def gcs_dag_location(self):
    if not self._gcs_dag_location:
      environment_details = Api.environment_details_get(self._zone, self._environment)

      if ('config' not in environment_details or
              'gcsDagLocation' not in environment_details.get('config')):
        raise ValueError('Dag location unavailable from Composer environment {0}'.format(
          self._environment))
      gcs_dag_location = environment_details['config']['gcsDagLocation']

      if gcs_dag_location is None or not self.gcs_file_regexp.match(gcs_dag_location):
        raise ValueError(
          'Dag location {0} from Composer environment {1} is in incorrect format'.format(
            gcs_dag_location, self._environment))

      self._gcs_dag_location = gcs_dag_location
      if gcs_dag_location.endswith('/') is False:
        self._gcs_dag_location = self._gcs_dag_location + '/'

    return self._gcs_dag_location
