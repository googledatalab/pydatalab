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
import google.datalab.bigquery as bigquery
from google.datalab import utils


class Composer(object):
  """ Represents a Composer object that encapsulates a set of functionality relating to the
  Cloud Composer service.

  This object can be used to generate the python airflow spec.
  """

  def __init__(self, context, zone, environment):
    """ Initializes an instance of a Composer object.

    Args:
      context: Pydatalab context object.
      zone: Zone in which Composer environment has been created.
      environment: Name of the Composer environment.
    """
    self._context = context
    self._zone = zone
    self._environment = environment

  def deploy(self, dag_string):
    client = gcs.Client()
    bucket = client.get_bucket(self.bucket_name)
    filename = 'dags/{0}.py'.format(self._name)
    blob = gcs.Blob(filename, bucket)
    blob.upload_from_string(dag_string)

  @property
  def bucket_name(self):
    return 'airflow-staging-test36490808-bucket'