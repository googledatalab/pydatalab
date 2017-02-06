# Copyright 2015 Google Inc. All rights reserved.
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

"""Miscellaneous simple utility functions."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from builtins import str

try:
    import http.client as httplib
except ImportError:
    import httplib

import pytz
import subprocess
import socket
import traceback
import types
import os
import json
import oauth2client.client


def print_exception_with_last_stack(e):
  """ Print the call stack of the last exception plu sprint the passed exception.

  Args:
    e: the exception to print.
  """
  traceback.print_exc()
  print(str(e))


def get_item(env, name, default=None):
  """ Get an item from a dictionary, handling nested lookups with dotted notation.

  Args:
    env: the environment (dictionary) to use to look up the name.
    name: the name to look up, in dotted notation.
    default: the value to return if the name if not found.

  Returns:
    The result of looking up the name, if found; else the default.
  """
  # TODO: handle attributes
  for key in name.split('.'):
    if isinstance(env, dict) and key in env:
      env = env[key]
    elif isinstance(env, types.ModuleType) and key in env.__dict__:
      env = env.__dict__[key]
    else:
      return default
  return env


def compare_datetimes(d1, d2):
  """ Compares two datetimes safely, whether they are timezone-naive or timezone-aware.

  If either datetime is naive it is converted to an aware datetime assuming UTC.

  Args:
    d1: first datetime.
    d2: second datetime.

  Returns:
    -1 if d1 < d2, 0 if they are the same, or +1 is d1 > d2.
  """
  if d1.tzinfo is None or d1.tzinfo.utcoffset(d1) is None:
    d1 = d1.replace(tzinfo=pytz.UTC)
  if d2.tzinfo is None or d2.tzinfo.utcoffset(d2) is None:
    d2 = d2.replace(tzinfo=pytz.UTC)
  if d1 < d2:
    return -1
  elif d1 > d2:
    return 1
  return 0


def pick_unused_port():
  """ get an unused port on the VM.

  Returns:
    An unused port.
  """
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind(('localhost', 0))
  addr, port = s.getsockname()
  s.close()
  return port


def is_http_running_on(port):
  """ Check if an http server runs on a given port.

  Args:
    The port to check.
  Returns:
    True if it is used by an http server. False otherwise.
  """
  try:
    conn = httplib.HTTPConnection('127.0.0.1:' + str(port))
    conn.connect()
    conn.close()
    return True
  except Exception as e:
    return False


def gcs_copy_file(source, dest):
  """ Copy file from source to destination. The paths can be GCS or local.

  Args:
    source: the source file path.
    dest: the destination file path.
  """
  subprocess.check_call(['gsutil', '-q', 'cp', source, dest])


""" Support for getting gcloud credentials. """

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
    credentials = oauth2client.client.GoogleCredentials.get_application_default()
    if credentials.create_scoped_required():
      credentials = credentials.create_scoped(CREDENTIAL_SCOPES)
    return credentials
  except Exception as e:

    # Try load user creds from file
    cred_file = get_config_dir() + '/credentials'
    if os.path.exists(cred_file):
      with open(cred_file) as f:
        creds= json.loads(f.read())
      # Use the first gcloud one we find
      for entry in creds['data']:
        if entry['key']['type'] == 'google-cloud-sdk':
          return oauth2client.client.OAuth2Credentials.from_json(json.dumps(entry['credential']))

    if type(e) == oauth2client.client.ApplicationDefaultCredentialsError:
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


def get_default_project_id():
  """ Get default project id from config or environment var.

  Returns: the project id if available, or None.
  """
  # Try getting default project id from gcloud. If it fails try config.json.
  try:
    proc = subprocess.Popen(['gcloud', 'config', 'list', '--format', 'value(core.project)'],
                            stdout=subprocess.PIPE)
    stdout, _ = proc.communicate()
    value = stdout.strip()
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
