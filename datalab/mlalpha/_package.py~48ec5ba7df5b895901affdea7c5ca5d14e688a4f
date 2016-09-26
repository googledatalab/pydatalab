# Copyright 2016 Google Inc. All rights reserved.
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

"""Implements Tarball Packaging for CloudML Training Program."""


import os
import shutil
import subprocess
import tempfile


class Packager(object):
  """Represents a packager."""

  def package(self, files, package_name):
    """Package a list of file contents into a python tarball package.

    Args:
      files: A dictionary with key as module name and value as module contents. File names
             will be key + 'py'.
      package_name: the name of the package.

    Returns: the path of the created package, in a temp directory.
    """
    tempdir = tempfile.mkdtemp()
    trainer_dir = os.path.join(tempdir, package_name)
    os.mkdir(trainer_dir)
    files['__init__'] = ''
    for name, content in files.iteritems():
      file_path = os.path.join(trainer_dir, name + '.py')
      with open(file_path, 'w') as file:
        file.write(content)

    setup_py = os.path.join(tempdir, 'setup.py')
    content = """from setuptools import setup

setup(
    name='%s',
    version='0.1',
    packages=['%s'],
)""" % (package_name, package_name)
    with open(setup_py, 'w') as file:
      file.write(content)
    previous_cwd = os.getcwd()
    os.chdir(tempdir)
    sdist = ['python', setup_py, 'sdist', '--format=gztar', '-d', tempdir]
    subprocess.check_call(sdist)
    os.chdir(previous_cwd)
    shutil.rmtree(trainer_dir)
    os.remove(setup_py)
    return os.path.join(tempdir, '%s-0.1.tar.gz' % package_name)
