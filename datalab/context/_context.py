# Copyright 2014 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements Context functionality."""
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import object

from . import _project
from . import _utils


class Context(object):
  """Maintains contextual state for connecting to Cloud APIs.
  """

  _global_context = None

  def __init__(self, project_id, credentials):
    """Initializes an instance of a Context object.

    Args:
      project_id: the current cloud project.
      credentials: the credentials to use to authorize requests.
    """
    self._project_id = project_id
    self._credentials = credentials

  @property
  def credentials(self):
    """Retrieves the value of the credentials property.

    Returns:
      The current credentials used in authorizing API requests.
    """
    return self._credentials

  def set_credentials(self, credentials):
    """ Set the credentials for the context. """
    self._credentials = credentials

  @property
  def project_id(self):
    """Retrieves the value of the project_id property.

    Returns:
      The current project id to associate with API requests.
    """
    if not self._project_id:
      raise Exception('No project ID found. Perhaps you should set one by running '
                      '"%projects set <project-id>" in a code cell.')
    return self._project_id

  def set_project_id(self, project_id):
    """ Set the project_id for the context. """
    self._project_id = project_id
    if self == Context._global_context:
      try:
        from google.datalab import Context as new_context
        new_context.default().set_project_id(project_id)
      except ImportError:
        # If the new library is not loaded, then we have nothing to do.
        pass

  @staticmethod
  def is_signed_in():
    """ If the user has signed in or it is on GCE VM with default credential."""
    try:
      _utils.get_credentials()
      return True
    except Exception:
      return False

  @staticmethod
  def default():
    """Retrieves a default Context object, creating it if necessary.

      The default Context is a global shared instance used every time the default context is
      retrieved.

      Attempting to use a Context with no project_id will raise an exception, so on first use
      set_project_id must be called.

    Returns:
      An initialized and shared instance of a Context object.
    """
    credentials = _utils.get_credentials()
    if Context._global_context is None:
      project = _project.Projects.get_default_id(credentials)
      Context._global_context = Context(project, credentials)
    else:
      # Always update the credentials in case the access token is revoked or expired
      Context._global_context.set_credentials(credentials)
    return Context._global_context
