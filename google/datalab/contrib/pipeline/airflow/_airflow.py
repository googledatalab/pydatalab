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

import google.datalab.storage as storage


class Airflow(object):
  """ Represents a Airflow object that encapsulates a set of functionality relating to the
  Cloud Airflow service.

  This object can be used to generate the python airflow spec.
  """

  def __init__(self, gcs_dag_bucket, gcs_dag_file_path=None):
    """ Initializes an instance of a Airflow object.

    Args:
      gcs_dag_bucket: Bucket where Airflow expects dag files to be uploaded.
      gcs_dag_file_path: File path of the Airflow dag files.
    """
    self._gcs_dag_bucket = gcs_dag_bucket
    self._gcs_dag_file_path = gcs_dag_file_path or ''

  def deploy(self, name, dag_string):
    if self._gcs_dag_file_path is not '' and self._gcs_dag_file_path.endswith('/') is False:
      self._gcs_dag_file_path = self._gcs_dag_file_path + '/'
    file_name = '{0}{1}.py'.format(self._gcs_dag_file_path, name)

    bucket = storage.Bucket(self._gcs_dag_bucket)
    file_object = bucket.object(file_name)
    file_object.write_stream(dag_string, 'text/plain')
