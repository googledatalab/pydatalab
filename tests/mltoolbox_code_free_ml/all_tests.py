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
import unittest

class RunTestScript(unittest.TestCase):
  """Makes a subprocess call to the test script.

  Note that we cannot simply install the mltoolbox package and run each step
  as it is designed to run as a sample, not as a moduel. We run the test script
  as a subprocess to 
  """
  def __init__(self, *args, **kwargs):
    super(RunTestScript, self).__init__(*args, **kwargs)

    # Path to the test script.
    self._root_path = os.path.abspath(
	    	os.path.join(
	    			os.path.dirname(__file__),
    		    '..', '..', 'solutionbox', 'code_free_ml', 'test_mltoolbox'))

  def test_local(self):
  	"""Test code_free_ml without ML Engine.

  	One test does use BigQuery.
  	"""
  	cmd = 'bash %s' % os.path.join(self._root_path, 'run_all.sh')
  	subprocess.check_call(cmd, cwd=self._root_path, shell=True)

