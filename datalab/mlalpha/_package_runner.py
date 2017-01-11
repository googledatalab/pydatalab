# Copyright 2017 Google Inc. All rights reserved.
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

"""Implements running Datalab ML Solution Packages."""

import inspect
import google.cloud.ml as ml
import os
import shutil
import subprocess
import sys
import tempfile


PACKAGE_NAMESPACE = 'datalab_solutions.'

class PackageRunner(object):
  """A Helper class to run Datalab ML solution packages."""

  def __init__(self, package_uri):
    """
    Args:
      package_uri: The uri of the package. The file base name needs to be in the form of
          "name-version", such as "inception-0.1". The first part split by "-" will be used
          as the last part of the namespace. In the example above,
          "datalab_solutions.inception" will be the namespace.
    """
    self._package_uri = package_uri
    self._name = os.path.basename(package_uri).split('-')[0]
    
  def _install_to_temp(self):
    install_dir = tempfile.mkdtemp()
    tar_path = self._package_uri
    if tar_path.startswith('gs://'):
      tar_path = os.path.join(install_dir, os.path.basename(tar_path))
      ml.util._file.copy_file(self._package_uri, tar_path)
    subprocess.check_call(['pip', 'install', tar_path, '--target', install_dir,
                           '--upgrade', '--force-reinstall'])
    sys.path.insert(0, install_dir)
    return install_dir

  def _cleanup_installation(self, install_dir):
    if sys.path[0] == install_dir:
      del sys.path[0]
    shutil.rmtree(install_dir)
    
  def get_func_args_and_docstring(self, func_name):
    """Get function args and docstrings.
    Args:
      func_name: name of the function.
    Returns:
      A tuple of function argspec, function docstring.
    """
    install_dir = self._install_to_temp()
    try:
      func = getattr(__import__(PACKAGE_NAMESPACE + self._name, fromlist=[func_name]), func_name)
      return inspect.getargspec(func), func.__doc__
    finally:
      self._cleanup_installation(install_dir)    

  def run_func(self, func_name, args):
    """Run a function.
    Args:
      func_name: name of the function.
      args: args supplied to the functions.
    Returns:
      function return values.
    """
    install_dir = self._install_to_temp()
    try:
      func = getattr(__import__(PACKAGE_NAMESPACE + self._name, fromlist=[func_name]), func_name)
      ret = func(**args)
      return ret
    finally:
      self._cleanup_installation(install_dir)
