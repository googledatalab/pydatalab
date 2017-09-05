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
"""Integration Tests of PredictionExplainer."""

from __future__ import absolute_import
from __future__ import unicode_literals
import unittest
import google.datalab
from PIL import Image
import numpy as np
import os
import shutil
import six
import tempfile


# import Python so we can mock the parts we need to here.
import IPython.core.display
import IPython.core.magic

from google.datalab.contrib.mlworkbench import PredictionExplainer


# Some tests put files in GCS or use BigQuery. If HAS_CREDENTIALS is false,
# those tests will not run.
HAS_CREDENTIALS = True
try:
  google.datalab.Context.default().project_id
except Exception:
  HAS_CREDENTIALS = False


def noop_decorator(func):
  return func


IPython.core.magic.register_line_cell_magic = noop_decorator
IPython.core.magic.register_line_magic = noop_decorator
IPython.core.magic.register_cell_magic = noop_decorator
IPython.core.display.HTML = lambda x: x
IPython.core.display.JSON = lambda x: x

import google.datalab.contrib.mlworkbench.commands._ml as mlmagic  # noqa


class TestMLExplainer(unittest.TestCase):
  """Integration tests of PredictionExplainer"""

  def setUp(self):
    self._code_path = mlmagic.MLTOOLBOX_CODE_PATH
    mlmagic.MLTOOLBOX_CODE_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     '../../solutionbox/code_free_ml/mltoolbox/code_free_ml'))
    self._test_dir = tempfile.mkdtemp()

  def tearDown(self):
    mlmagic.MLTOOLBOX_CODE_PATH = self._code_path
    shutil.rmtree(self._test_dir)

  def _create_text_test_data(self):
    """Create text model."""

    test_data = """1,sour green round,lime
    2,melon green long,cucumber
    3,sweet round red,apple"""
    train_csv = os.path.join(self._test_dir, 'train.csv')
    with open(train_csv, 'w') as f:
      f.write(test_data)

    analyze_dir = os.path.join(self._test_dir, 'analysistxt')
    train_dir = os.path.join(self._test_dir, 'traintxt')

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

  def _create_image_test_data(self):
    image_path1 = os.path.join(self._test_dir, 'img1.jpg')
    image_path2 = os.path.join(self._test_dir, 'img2.jpg')
    image_path3 = os.path.join(self._test_dir, 'img3.jpg')
    Image.new('RGBA', size=(128, 128), color=(155, 211, 64)).save(image_path1, "JPEG")
    Image.new('RGB', size=(64, 64), color=(111, 21, 86)).save(image_path2, "JPEG")
    Image.new('RGBA', size=(16, 16), color=(255, 21, 1)).save(image_path3, "JPEG")
    test_data = """1,1.2,word1 word2,%s,true
2,3.2,word2 word3,%s,false
5,-2.1,word3 word4,%s,true""" % (image_path1, image_path2, image_path3)

    train_csv = os.path.join(self._test_dir, 'train.csv')
    with open(train_csv, 'w') as f:
      f.write(test_data)

    analyze_dir = os.path.join(self._test_dir, 'analysisimg')
    transform_dir = os.path.join(self._test_dir, 'transformimg')
    train_dir = os.path.join(self._test_dir, 'trainimg')

    mlmagic.ml(
        line='analyze',
        cell="""\
            output: %s
            training_data:
              csv: %s
              schema:
                - name: key
                  type: INTEGER
                - name: num
                  type: FLOAT
                - name: text
                  type: STRING
                - name: img_url
                  type: STRING
                - name: target
                  type: STRING
            features:
              key:
                transform: key
              num:
                transform: scale
              text:
                transform: bag_of_words
              img_url:
                transform: image_to_vec
              target:
                transform: target""" % (analyze_dir, train_csv))

    mlmagic.ml(
        line='transform',
        cell="""\
            output: %s
            analysis: %s
            prefix: train
            training_data:
              csv: %s""" % (transform_dir, analyze_dir, train_csv))

    mlmagic.ml(
        line='train',
        cell="""\
            output: %s
            analysis: %s
            training_data:
              transformed: %s/train-*
            evaluation_data:
              transformed: %s/train-*
            model_args:
              model: linear_classification
              top-n: 0
              max-steps: 200""" % (train_dir, analyze_dir, transform_dir, transform_dir))

  @unittest.skipIf(not six.PY2 or not HAS_CREDENTIALS,
                   'Integration test that invokes mlworkbench with DataFlow.')
  def test_text_explainer(self):
    """Test text explainer."""

    self._create_text_test_data()
    explainer = PredictionExplainer(os.path.join(self._test_dir, 'traintxt', 'model'))
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

  @unittest.skipIf(not six.PY2, 'Integration test that invokes mlworkbench with DataFlow.')
  def test_image_explainer(self):
    """Test image explainer."""

    self._create_image_test_data()
    explainer = PredictionExplainer(os.path.join(self._test_dir, 'trainimg', 'model'))
    exp_instance = explainer.explain_image(
        ['true', 'false'],
        '4,2.0,word2 word1,%s' % os.path.join(self._test_dir, 'img1.jpg'),
        num_samples=50)

    for i in range(2):
      image, mask = exp_instance.get_image_and_mask(i, positive_only=False, num_features=3,
                                                    )
      # image's dimension is length*width*channel
      self.assertEqual(len(np.asarray(image).shape), 3)
      # mask's dimension is length*width
      self.assertEqual(len(np.asarray(mask).shape), 2)
