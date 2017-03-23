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
"""Test the datalab interface functions.

Note that the calls to do analysis, training, and prediction are all mocked.
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# Set up the path so that we can import local packages.
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),
                    '../../solutionbox/structured_data/')))  # noqa

from test_mltoolbox.test_package_functions import *  # noqa
