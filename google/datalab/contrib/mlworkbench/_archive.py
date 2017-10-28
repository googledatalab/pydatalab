# Copyright 2017 Google Inc. All rights reserved.
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

"""Google Cloud Platform library - ml cell magic."""
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import shutil
import tempfile
import tensorflow as tf

import google.datalab.contrib.mlworkbench._shell_process as _shell_process


def extract_archive(archive_path, dest):
  """Extract a local or GCS archive file to a folder.

  Args:
    archive_path: local or gcs path to a *.tar.gz or *.tar file
    dest: local folder the archive will be extracted to
  """
  # Make the dest folder if it does not exist
  if not os.path.isdir(dest):
    os.makedirs(dest)

  try:
    tmpfolder = None

    if (not tf.gfile.Exists(archive_path)) or tf.gfile.IsDirectory(archive_path):
      raise ValueError('archive path %s is not a file' % archive_path)

    if archive_path.startswith('gs://'):
      # Copy the file to a local temp folder
      tmpfolder = tempfile.mkdtemp()
      cmd_args = ['gsutil', 'cp', archive_path, tmpfolder]
      _shell_process.run_and_monitor(cmd_args, os.getpid())
      archive_path = os.path.join(tmpfolder, os.path.name(archive_path))

    if archive_path.lower().endswith('.tar.gz'):
      flags = '-xzf'
    elif archive_path.lower().endswith('.tar'):
      flags = '-xf'
    else:
      raise ValueError('Only tar.gz or tar.Z files are supported.')

    cmd_args = ['tar', flags, archive_path, '-C', dest]
    _shell_process.run_and_monitor(cmd_args, os.getpid())
  finally:
    if tmpfolder:
      shutil.rmtree(tmpfolder)
