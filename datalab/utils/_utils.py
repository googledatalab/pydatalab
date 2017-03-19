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
  except Exception:
    return False


def gcs_copy_file(source, dest):
  """ Copy file from source to destination. The paths can be GCS or local.

  Args:
    source: the source file path.
    dest: the destination file path.
  """
  subprocess.check_call(['gsutil', '-q', 'cp', source, dest])
