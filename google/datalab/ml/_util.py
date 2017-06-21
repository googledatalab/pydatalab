# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from googleapiclient import discovery

import os
import shutil
import subprocess
import tempfile
import tensorflow as tf
import time

import datalab.context
import google.datalab.utils as dlutils


# TODO: Create an Operation class.
def wait_for_long_running_operation(operation_full_name):
  print('Waiting for operation "%s"' % operation_full_name)
  api = discovery.build('ml', 'v1', credentials=datalab.context.Context.default().credentials)
  while True:
    response = api.projects().operations().get(name=operation_full_name).execute()
    if 'done' not in response or response['done'] is not True:
      time.sleep(3)
    else:
      if 'error' in response:
        print(response['error'])
      else:
        print('Done.')
      break


def package_and_copy(package_root_dir, setup_py, output_tar_path):
  """Repackage an CloudML package and copy it to a staging dir.

  Args:
    package_root_dir: the root dir to install package from. Usually you can get the path
        from inside your module using a relative path to __file__.
    setup_py: the path to setup.py.
    output_tar_path: the GCS path of the output tarball package.
  Raises:
    ValueError if output_tar_path is not a GCS path, or setup_py does not exist.
  """
  if not output_tar_path.startswith('gs://'):
    raise ValueError('output_tar_path needs to be a GCS path.')
  if not os.path.isfile(setup_py):
    raise ValueError('Supplied file "%s" does not exist.' % setup_py)

  dest_setup_py = os.path.join(package_root_dir, 'setup.py')
  if dest_setup_py != setup_py:
    # setuptools requires a "setup.py" in the current dir, so copy setup.py there.
    # Also check if there is an existing setup.py. If so, back it up.
    if os.path.isfile(dest_setup_py):
      os.rename(dest_setup_py, dest_setup_py + '._bak_')
    shutil.copyfile(setup_py, dest_setup_py)

  tempdir = tempfile.mkdtemp()
  previous_cwd = os.getcwd()
  os.chdir(package_root_dir)
  try:
    # Repackage.
    sdist = ['python', dest_setup_py, 'sdist', '--format=gztar', '-d', tempdir]
    subprocess.check_call(sdist)

    # Copy to GCS.
    source = os.path.join(tempdir, '*.tar.gz')
    gscopy = ['gsutil', 'cp', source, output_tar_path]
    subprocess.check_call(gscopy)
    return
  finally:
    os.chdir(previous_cwd)
    if dest_setup_py != setup_py:
      os.remove(dest_setup_py)
    if os.path.isfile(dest_setup_py + '._bak_'):
      os.rename(dest_setup_py + '._bak_', dest_setup_py)
    shutil.rmtree(tempdir)


def read_file_to_string(path):
  """Read a file into a string."""
  bytes_string = tf.gfile.Open(path, 'r').read()
  return dlutils.python_portable_string(bytes_string)


def open_local_or_gcs(path, mode):
  """Opens the given path."""
  return tf.gfile.Open(path, mode)


def glob_files(path):
  """Glob the given path."""
  return tf.gfile.Glob(path)
