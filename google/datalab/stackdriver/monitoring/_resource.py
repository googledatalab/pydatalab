# Copyright 2016 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.  See the License for the specific language governing permissions and limitations under
# the License.

"""Provides the ResourceDescriptors in the monitoring API."""

from __future__ import absolute_import
from builtins import object

import fnmatch
import pandas

from . import _utils


class ResourceDescriptors(object):
  """ResourceDescriptors object for retrieving the resource descriptors."""

  _DISPLAY_HEADERS = ('Resource type', 'Display name', 'Labels')

  def __init__(self, filter_string=None, context=None):
    """Initializes the ResourceDescriptors based on the specified filters.

    Args:
      filter_string: An optional filter expression describing the resource
          descriptors to be returned.
      context: An optional Context object to use instead of the global default.
    """
    self._client = _utils.make_client(context)
    self._filter_string = filter_string
    self._descriptors = None

  def list(self, pattern='*'):
    """Returns a list of resource descriptors that match the filters.

    Args:
      pattern: An optional pattern to further filter the descriptors. This can
          include Unix shell-style wildcards. E.g. ``"aws*"``, ``"*cluster*"``.

    Returns:
      A list of ResourceDescriptor objects that match the filters.
    """
    if self._descriptors is None:
      self._descriptors = self._client.list_resource_descriptors(
          filter_string=self._filter_string)
    return [resource for resource in self._descriptors
            if fnmatch.fnmatch(resource.type, pattern)]

  def as_dataframe(self, pattern='*', max_rows=None):
    """Creates a pandas dataframe from the descriptors that match the filters.

    Args:
      pattern: An optional pattern to further filter the descriptors. This can
          include Unix shell-style wildcards. E.g. ``"aws*"``, ``"*cluster*"``.
      max_rows: The maximum number of descriptors to return. If None, return
          all.

    Returns:
      A pandas dataframe containing matching resource descriptors.
    """
    data = []
    for i, resource in enumerate(self.list(pattern)):
      if max_rows is not None and i >= max_rows:
        break
      labels = ', '. join([l.key for l in resource.labels])
      data.append([resource.type, resource.display_name, labels])

    return pandas.DataFrame(data, columns=self._DISPLAY_HEADERS)
