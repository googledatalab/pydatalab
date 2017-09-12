# Copyright 2016 Google Inc. All rights reserved.
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

""" Support for getting gcloud credentials. """

from __future__ import absolute_import
from __future__ import unicode_literals
import json
import os
import subprocess

import oauth2client.client
import google.auth
import google.auth.exceptions
import google.auth.credentials
import google.auth._oauth2client


# TODO(ojarjur): This limits the APIs against which Datalab can be called
# (when using a service account with a credentials file) to only being those
# that are part of the Google Cloud Platform. We should either extend this
# to all of the API scopes that Google supports, or make it extensible so
# that the user can define for themselves which scopes they want to use.
CREDENTIAL_SCOPES = [
  'https://www.googleapis.com/auth/cloud-platform',
]


def _in_datalab_docker():
  return os.path.exists('/datalab') and os.getenv('DATALAB_ENV')


def get_config_dir():
  config_dir = os.getenv('CLOUDSDK_CONFIG')
  if config_dir is None:
    if os.name == 'nt':
      try:
        config_dir = os.path.join(os.environ['APPDATA'], 'gcloud')
      except KeyError:
        # This should never happen unless someone is really messing with things.
        drive = os.environ.get('SystemDrive', 'C:')
        config_dir = os.path.join(drive, '\\gcloud')
    else:
      config_dir = os.path.join(os.path.expanduser('~'), '.config/gcloud')
  return config_dir


def _convert_oauth2client_creds(credentials):
  new_credentials = google.oauth2.credentials.Credentials(
    token=credentials.access_token,
    refresh_token=credentials.refresh_token,
    token_uri=credentials.token_uri,
    client_id=credentials.client_id,
    client_secret=credentials.client_secret,
    scopes=credentials.scopes)

  new_credentials._expires = credentials.token_expiry
  return new_credentials


def get_credentials():
  """ Get the credentials to use. We try application credentials first, followed by
      user credentials. The path to the application credentials can be overridden
      by pointing the GOOGLE_APPLICATION_CREDENTIALS environment variable to some file;
      the path to the user credentials can be overridden by pointing the CLOUDSDK_CONFIG
      environment variable to some directory (after which we will look for the file
      $CLOUDSDK_CONFIG/gcloud/credentials). Unless you have specific reasons for
      overriding these the defaults should suffice.
  """
  try:
    credentials, _ = google.auth.default()
    credentials = google.auth.credentials.with_scopes_if_required(credentials, CREDENTIAL_SCOPES)
    return credentials
  except Exception as e:

    # Try load user creds from file
    cred_file = get_config_dir() + '/credentials'
    if os.path.exists(cred_file):
      with open(cred_file) as f:
        creds = json.loads(f.read())
      # Use the first gcloud one we find
      for entry in creds['data']:
        if entry['key']['type'] == 'google-cloud-sdk':
          creds = oauth2client.client.OAuth2Credentials.from_json(json.dumps(entry['credential']))
          return _convert_oauth2client_creds(creds)

    if type(e) == google.auth.exceptions.DefaultCredentialsError:
      # If we are in Datalab container, change the message to be about signing in.
      if _in_datalab_docker():
        raise Exception('No application credentials found. Perhaps you should sign in.')

    raise e


def save_project_id(project_id):
  """ Save project id to config file.

  Args:
    project_id: the project_id to save.
  """
  # Try gcloud first. If gcloud fails (probably because it does not exist), then
  # write to a config file.
  try:
    subprocess.call(['gcloud', 'config', 'set', 'project', project_id])
  except:
    config_file = os.path.join(get_config_dir(), 'config.json')
    config = {}
    if os.path.exists(config_file):
      with open(config_file) as f:
        config = json.loads(f.read())
    config['project_id'] = project_id
    with open(config_file, 'w') as f:
      f.write(json.dumps(config))


def get_project_id():
  """ Get default project id from config or environment var.

  Returns: the project id if available, or None.
  """
  # Try getting default project id from gcloud. If it fails try config.json.
  try:
    proc = subprocess.Popen(['gcloud', 'config', 'list', '--format', 'value(core.project)'],
                            stdout=subprocess.PIPE)
    stdout, _ = proc.communicate()
    value = stdout.decode().strip()
    if proc.poll() == 0 and value:
      return value
  except:
    pass

  config_file = os.path.join(get_config_dir(), 'config.json')
  if os.path.exists(config_file):
    with open(config_file) as f:
      config = json.loads(f.read())
      if 'project_id' in config and config['project_id']:
        return str(config['project_id'])

  if os.getenv('PROJECT_ID') is not None:
    return os.getenv('PROJECT_ID')
  return None
