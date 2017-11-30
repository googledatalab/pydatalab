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
"""Test the code_free_ml package
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import subprocess
import six
import unittest


class RunTestScript(unittest.TestCase):
  """Makes a subprocess call to the test script.

  Note that we cannot simply install the mltoolbox package and run each step
  as it is designed to run as a sample/script, not as a module.
  """
  def __init__(self, *args, **kwargs):
    super(RunTestScript, self).__init__(*args, **kwargs)

    # Path to the test script.
    self._root_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__),
                        '..', '..', 'solutionbox', 'ml_workbench', 'test_tensorflow')),
        os.path.abspath(os.path.join(os.path.dirname(__file__),
                        '..', '..', 'solutionbox', 'ml_workbench', 'test_xgboost')),
    ]

  @unittest.skipIf(not six.PY2, 'Python 2 is required')
  def test_local(self):
    """Run some of the code_free_ml tests.

    Tests that use GCS services like GCS or BigQuery are not ran.
    """
    for root_path in self._root_paths:
      cmd = 'bash %s' % os.path.join(root_path, 'run_all.sh')
      subprocess.check_call(cmd, cwd=root_path, shell=True)
