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

from google.datalab.utils import _utils as du


class Context(object):
  """Maintains contextual state for connecting to Cloud APIs.
  """

  _global_context = None

  def __init__(self, project_id, credentials, config=None):
    """Initializes an instance of a Context object.

    Args:
      project_id: the current cloud project.
      credentials: the credentials to use to authorize requests.
      config: key/value configurations for cloud operations
    """
    self._project_id = project_id
    self._credentials = credentials
    self._config = config if config is not None else Context._get_default_config()

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
      raise Exception('No project ID found. Perhaps you should set one by running'
                      '"%datalab project set -p <project-id>" in a code cell.')
    return self._project_id

  def set_project_id(self, project_id):
    """ Set the project_id for the context. """
    self._project_id = project_id
    if self == Context._global_context:
      du.save_project_id(self._project_id)

  @property
  def config(self):
    """ Retrieves the value of the config property.

    Returns:
      The current config object used in cloud operations
    """
    return self._config

  def set_config(self, config):
    """ Set the config property for the context. """
    self._config = config

  @staticmethod
  def _is_signed_in():
    """ If the user has signed in or it is on GCE VM with default credential."""
    try:
      du.get_credentials()
      return True
    except Exception:
      return False

  @staticmethod
  def _get_default_config():
    """Return a default config object"""
    return {
      'bigquery_billing_tier': None
    }

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
    credentials = du.get_credentials()
    project = du.get_default_project_id()
    if Context._global_context is None:
      config = Context._get_default_config()
      Context._global_context = Context(project, credentials, config)
    else:
      # Always update everything in case the access token is revoked or expired, config changed,
      # or project changed.
      Context._global_context.set_credentials(credentials)
      Context._global_context.set_project_id(project)
    return Context._global_context
