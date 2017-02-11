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

import datetime
import os
import shutil
import subprocess
import tempfile


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
    os.remove(dest_setup_py)
    if os.path.isfile(dest_setup_py + '._bak_'):
      os.rename(dest_setup_py + '._bak_', dest_setup_py)
    shutil.rmtree(tempdir)
