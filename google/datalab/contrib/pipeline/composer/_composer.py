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


class Composer(object):
  """ Represents a Composer object that encapsulates a set of functionality relating to the
  Cloud Composer service.

  This object can be used to generate the python airflow spec.
  """

  def __init__(self, zone, environment):
    """ Initializes an instance of a Composer object.

    Args:
      zone: Zone in which Composer environment has been created.
      environment: Name of the Composer environment.
    """
    self._zone = zone
    self._environment = environment

  def deploy(self, name, dag_string):
    client = gcs.Client()
    bucket = client.get_bucket(self.bucket_name)
    filename = 'dags/{0}.py'.format(name)
    blob = gcs.Blob(filename, bucket)
    blob.upload_from_string(dag_string)

  @property
  def bucket_name(self):
    # TODO(rajivpb): Get this programmatically from the Composer API
    return 'airflow-staging-test36490808-bucket'

  @property
  def get_bucket_name(self):
    # environment_details = Api().environment_details_get(self._zone, self._environment)

    # TODO(rajivpb): Get this programmatically from the Composer API
    return 'airflow-staging-test36490808-bucket'
