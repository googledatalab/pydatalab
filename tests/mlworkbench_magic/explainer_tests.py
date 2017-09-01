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
"""Tests the \%\%ml magics functions without runing any jobs."""

from __future__ import absolute_import
from __future__ import unicode_literals
import unittest
import os
import shutil
import tempfile


# import Python so we can mock the parts we need to here.
import IPython.core.display
import IPython.core.magic

import google.datalab.contrib.mlworkbench.commands._ml as mlmagic  # noqa
from google.datalab.contrib.mlworkbench import PredictionExplainer


def noop_decorator(func):
  return func


IPython.core.magic.register_line_cell_magic = noop_decorator
IPython.core.magic.register_line_magic = noop_decorator
IPython.core.magic.register_cell_magic = noop_decorator
IPython.core.display.HTML = lambda x: x
IPython.core.display.JSON = lambda x: x


mlmagic.MLTOOLBOX_CODE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../solutionbox/code_free_ml/mltoolbox/code_free_ml'))


class TestMLExplainer(unittest.TestCase):
  """Integration tests of PredictionExplainer"""

  def setUp(self):
    self._test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self._test_dir)

  def _create_text_test_data(self):
    """Create text model."""

    test_data = """1,sour green round,lime
    2,melon green long,cucumber
    3,sweet round red,apple"""
    train_csv = os.path.join(self._test_dir, 'train.csv')
    with open(train_csv, 'w') as f:
      f.write(test_data)

    analyze_dir = os.path.join(self._test_dir, 'analysis')
    train_dir = os.path.join(self._test_dir, 'train')

    mlmagic.ml(
        line='analyze',
        cell="""\
            output: %s
            training_data:
              csv: %s
              schema:
                - name: key
                  type: INTEGER
                - name: text
                  type: STRING
                - name: target
                  type: STRING
            features:
              key:
                transform: key
              text:
                transform: bag_of_words
              target:
                transform: target""" % (analyze_dir, train_csv))

    mlmagic.ml(
        line='train',
        cell="""\
            output: %s
            analysis: %s
            training_data:
              csv: %s
            evaluation_data:
              csv: %s
            model_args:
              model: linear_classification
              top-n: 0
              max-steps: 300""" % (train_dir, analyze_dir, train_csv, train_csv))

  def test_text_explainer(self):
    """Test text explainer."""

    self._create_text_test_data()
    explainer = PredictionExplainer(os.path.join(self._test_dir, 'train', 'model'))
    exp_instance = explainer.explain_text(['apple', 'lime', 'cucumber'], '4,green long')
    apple = exp_instance.as_list(label=0)
    self.assertEqual(len(apple), 2)
    for word, score in apple:
      # "green" and "long" are both negative to "apple"
      self.assertLess(score, 0.0)

    cucumber = exp_instance.as_list(label=2)
    self.assertEqual(len(cucumber), 2)
    for word, score in cucumber:
      # "green" and "long" are both positive to "cucumber"
      self.assertGreater(score, 0.0)
