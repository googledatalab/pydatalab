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

"""Google Cloud Platform library - BigQuery SQL Dialect"""
from __future__ import absolute_import


class Dialect(object):
  """
  Represents the default BigQuery SQL dialect
  """
  _global_dialect = None

  def __init__(self, bq_dialect):
    self._global_dialect = bq_dialect

  @property
  def bq_dialect(self):
    """Retrieves the value of the bq_dialect property.

    Returns:
      The default BigQuery SQL dialect
    """
    return self._global_dialect

  def set_bq_dialect(self, bq_dialect):
    """ Set the default BigQuery SQL dialect"""
    if bq_dialect in ['legacy', 'standard']:
      self._global_dialect = bq_dialect

  @staticmethod
  def default():
    """Retrieves the default BigQuery SQL dialect, creating it if necessary.

    Returns:
      An initialized and shared instance of a Dialect object.
    """
    if Dialect._global_dialect is None:
      Dialect._global_dialect = Dialect('legacy')
    return Dialect._global_dialect
