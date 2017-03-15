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
from __future__ import absolute_import

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../solutionbox/structured_data/')))
import test_mltoolbox.test_sd_trainer as sdtraining


class TestCoreTrainingLib(sdtraining.TestTrainer):
  """Wraps the training tests in the structured data package.

  Four problems/models are built:
  	regression + dnn model
  	regression + linear model
  	classification + dnn model
  	classification + linear model

  These tests take about 30 seconds to run.
  """

  def __init__(self, *args, **kwargs):
    super(TestCoreTrainingLib, self).__init__(*args, **kwargs)

    # Test that training works, not that the model is good.
    self._max_steps = 50  
    self._check_model_loss = False

    # Don't print anything. Set to false for debugging.
    self._silent_output = True
  