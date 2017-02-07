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


import os
import shutil
import subprocess
import tempfile


# Keep it in sync with
# https://github.com/googledatalab/datalab/blob/master/containers/base/Dockerfile
DATALAB_SOLUTION_LIB_DIR = '/datalab/lib/pydatalab/'


def package_and_copy(package_name, out_dir):
  """Repackage an CloudML package and copy it to a staging dir.

  Args:
    package_name: the name of the package, such as 'inception'.
    out_dir: the ourput directory. Has to be GCS path.
  Returns:
    The destination package GS URL.
  Raises:
    ValueError if output_dir is not a GCS path.
  """
  if not out_dir.startswith('gs://'):
    raise ValueError('Output needs to be a GCS path.')

  package_dir = os.path.join(DATALAB_SOLUTION_LIB_DIR, 'solutionbox', package_name)
  tempdir = tempfile.mkdtemp()
  previous_cwd = os.getcwd()
  os.chdir(package_dir)
  try:
    # Repackage. In Datalab the source files of pydatalab is kept.
    setup_py = os.path.join(package_dir, 'setup.py')
    sdist = ['python', setup_py, 'sdist', '--format=gztar', '-d', tempdir]
    subprocess.check_call(sdist)

    # Copy to GCS.
    dest = os.path.join(out_dir, 'staging', package_name + '.tar.gz')
    source = os.path.join(tempdir, '*.tar.gz')
    gscopy = ['gsutil', 'cp', source, dest]
    subprocess.check_call(gscopy)
    return dest
  finally:
    os.chdir(previous_cwd)
    shutil.rmtree(tempdir)
