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

"""Groups for the Google Monitoring API."""

from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import object

import collections
import fnmatch

import pandas

import datalab.context

from . import _utils


class Groups(object):
  """Represents a list of Stackdriver groups for a project."""

  _DISPLAY_HEADERS = ('Group ID', 'Group name', 'Parent ID', 'Parent name',
                      'Is cluster', 'Filter')

  def __init__(self, project_id=None, context=None):
    """Initializes the Groups for a Stackdriver project.

    Args:
      project_id: An optional project ID or number to override the one provided
          by the context.
      context: An optional Context object to use instead of the global default.
    """
    self._context = context or datalab.context.Context.default()
    self._project_id = project_id or self._context.project_id
    self._client = _utils.make_client(project_id, context)
    self._group_dict = None

  def list(self, pattern='*'):
    """Returns a list of groups that match the filters.

    Args:
      pattern: An optional pattern to filter the groups based on their display
          name. This can include Unix shell-style wildcards. E.g.
          ``"Production*"``.

    Returns:
      A list of Group objects that match the filters.
    """
    if self._group_dict is None:
      self._group_dict = collections.OrderedDict(
          (group.id, group) for group in self._client.list_groups())

    return [group for group in self._group_dict.values()
            if fnmatch.fnmatch(group.display_name, pattern)]

  def as_dataframe(self, pattern='*', max_rows=None):
    """Creates a pandas dataframe from the groups that match the filters.

    Args:
      pattern: An optional pattern to further filter the groups. This can
          include Unix shell-style wildcards. E.g. ``"Production *"``,
          ``"*-backend"``.
      max_rows: The maximum number of groups to return. If None, return all.

    Returns:
      A pandas dataframe containing matching groups.
    """
    data = []
    for i, group in enumerate(self.list(pattern)):
      if max_rows is not None and i >= max_rows:
        break
      parent = self._group_dict.get(group.parent_id)
      parent_display_name = '' if parent is None else parent.display_name
      data.append([
          group.id, group.display_name, group.parent_id,
          parent_display_name, group.is_cluster, group.filter])

    return pandas.DataFrame(data, columns=self._DISPLAY_HEADERS)
