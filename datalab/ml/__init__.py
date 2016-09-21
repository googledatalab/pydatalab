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

"""CloudML Helper Library."""

from __future__ import absolute_import

from ._local_runner import LocalRunner
from ._cloud_runner import CloudRunner
from ._metadata import Metadata
from ._local_predictor import LocalPredictor
from ._cloud_predictor import CloudPredictor
from ._job import Jobs
from ._summary import Summary
from ._tensorboard import TensorBoardManager
from ._dataset import DataSet
from ._package import Packager
from ._cloud_models import CloudModels, CloudModelVersions
from ._confusion_matrix import ConfusionMatrix

from plotly.offline import init_notebook_mode

init_notebook_mode()

