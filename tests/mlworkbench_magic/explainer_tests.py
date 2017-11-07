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
from PIL import Image
import mock
import numpy as np
import os
import pandas as pd
import shutil
import six
from six.moves.urllib.request import urlopen
import tempfile


# import Python so we can mock the parts we need to here.
import IPython.core.display
import IPython.core.magic

from google.datalab.contrib.mlworkbench import PredictionExplainer


def noop_decorator(func):
  return func


IPython.core.magic.register_line_cell_magic = noop_decorator
IPython.core.magic.register_line_magic = noop_decorator
IPython.core.magic.register_cell_magic = noop_decorator
IPython.core.display.HTML = lambda x: x
IPython.core.display.JSON = lambda x: x
IPython.get_ipython = mock.Mock()
IPython.get_ipython().user_ns = {}

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

  def _create_tabular_test_data(self):
    """Create tabular model with text."""

    test_data = """1,5.0,monday,word1 word2 word3,true
    2,3.2,tuesday,word1 word3,true
    3,-1.1,friday,word1,false"""
    train_csv = os.path.join(self._test_dir, 'train.csv')
    with open(train_csv, 'w') as f:
      f.write(test_data)

    df = pd.read_csv(train_csv, names=['key', 'num', 'weekday', 'garbage', 'target'])
    analyze_dir = os.path.join(self._test_dir, 'analysistab')
    train_dir = os.path.join(self._test_dir, 'traintab')

    mlmagic.ml(
        line='dataset create',
        cell="""\
            format: csv
            name: mytabular
            schema:
                - name: key
                  type: INTEGER
                - name: num
                  type: FLOAT
                - name: weekday
                  type: STRING
                - name: garbage
                  type: STRING
                - name: target
                  type: STRING
            train: %s
            eval: %s""" % (train_csv, train_csv))

    mlmagic.ml(
        line='analyze',
        cell="""\
            output: %s
            data: mytabular
            features:
              key:
                transform: key
              num:
                transform: scale
              weekday:
                transform: one_hot
              garbage:
                transform: bag_of_words
              target:
                transform: target""" % (analyze_dir))

    mlmagic.ml(
        line='train',
        cell="""\
            output: %s
            analysis: %s
            data: mytabular
            notb: true
            model_args:
              model: linear_classification
              top-n: 0
              max-steps: 300""" % (train_dir, analyze_dir))
    return df

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
        line='dataset create',
        cell="""\
            format: csv
            name: mytext
            schema:
                - name: key
                  type: INTEGER
                - name: text
                  type: STRING
                - name: target
                  type: STRING
            train: %s
            eval: %s""" % (train_csv, train_csv))

    mlmagic.ml(
        line='analyze',
        cell="""\
            output: %s
            data: mytext
            features:
              key:
                transform: key
              text:
                transform: bag_of_words
              target:
                transform: target""" % (analyze_dir))

    mlmagic.ml(
        line='train',
        cell="""\
            output: %s
            analysis: %s
            data: mytext
            notb: true
            model_args:
              model: linear_classification
              top-n: 0
              max-steps: 300""" % (train_dir, analyze_dir))

  def _create_image_test_data(self):
    image_path1 = os.path.join(self._test_dir, 'img1.jpg')
    image_path2 = os.path.join(self._test_dir, 'img2.jpg')
    image_path3 = os.path.join(self._test_dir, 'img3.jpg')
    Image.new('RGB', size=(128, 128), color=(155, 211, 64)).save(image_path1, "JPEG")
    Image.new('RGB', size=(64, 64), color=(111, 21, 86)).save(image_path2, "JPEG")
    Image.new('RGB', size=(16, 16), color=(255, 21, 1)).save(image_path3, "JPEG")
    test_data = """1,1.2,word1 word2,%s,true
2,3.2,word2 word3,%s,false
5,-2.1,word3 word4,%s,true""" % (image_path1, image_path2, image_path3)

    train_csv = os.path.join(self._test_dir, 'train.csv')
    with open(train_csv, 'w') as f:
      f.write(test_data)

    analyze_dir = os.path.join(self._test_dir, 'analysisimg')
    transform_dir = os.path.join(self._test_dir, 'transformimg')
    train_dir = os.path.join(self._test_dir, 'trainimg')

    # Download inception checkpoint. Note that gs url doesn't work because
    # we may not have gcloud signed in when running the test.
    url = ('https://storage.googleapis.com/cloud-ml-data/img/' +
           'flower_photos/inception_v3_2016_08_28.ckpt')
    checkpoint_path = os.path.join(self._test_dir, "checkpoint")
    response = urlopen(url)
    with open(checkpoint_path, 'wb') as f:
      f.write(response.read())

    mlmagic.ml(
        line='dataset create',
        cell="""\
            format: csv
            name: myds
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
            train: %s
            eval: %s""" % (train_csv, train_csv))

    mlmagic.ml(
        line='analyze',
        cell="""\
            output: %s
            data: myds
            features:
              key:
                transform: key
              num:
                transform: scale
              text:
                transform: bag_of_words
              img_url:
                transform: image_to_vec
                checkpoint: %s
              target:
                transform: target""" % (analyze_dir, checkpoint_path))

    mlmagic.ml(
        line='transform',
        cell="""\
            output: %s
            analysis: %s
            data: myds""" % (transform_dir, analyze_dir))

    mlmagic.ml(
        line='dataset create',
        cell="""\
            format: transformed
            name: transformed_ds
            train: %s/train-*
            eval: %s/eval-*""" % (transform_dir, transform_dir))

    mlmagic.ml(
        line='train',
        cell="""\
            output: %s
            analysis: %s
            data: transformed_ds
            notb: true
            model_args:
              model: linear_classification
              top-n: 0
              max-steps: 200""" % (train_dir, analyze_dir))

  @unittest.skipIf(not six.PY2, 'Integration test that invokes mlworkbench with DataFlow.')
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
      image, mask = exp_instance.get_image_and_mask(i, positive_only=False, num_features=3)
      # image's dimension is length*width*channel
      self.assertEqual(len(np.asarray(image).shape), 3)
      # mask's dimension is length*width
      self.assertEqual(len(np.asarray(mask).shape), 2)

  @unittest.skipIf(not six.PY2, 'Integration test that invokes mlworkbench with DataFlow.')
  def test_image_prober(self):
    """Test image explainer."""

    self._create_image_test_data()
    explainer = PredictionExplainer(os.path.join(self._test_dir, 'trainimg', 'model'))
    raw_image, grads_vizs = explainer.probe_image(
        ['true', 'false'],
        '4,2.0,word2 word1,%s' % os.path.join(self._test_dir, 'img1.jpg'),
        top_percent=20)
    self.assertEqual((299, 299, 3), np.asarray(raw_image).shape)

    for im in grads_vizs:
      self.assertEqual((299, 299, 3), np.asarray(im).shape)
      arr = np.asarray(im)
      arr = arr.reshape(-1)
      self.assertGreater(float((arr == 0).sum()) / len(arr), 0.79)

  @unittest.skipIf(not six.PY2, 'Integration test that invokes mlworkbench with DataFlow.')
  def test_tabular_explainer(self):
    """Test tabular explainer."""

    train_df = self._create_tabular_test_data()

    explainer = PredictionExplainer(os.path.join(self._test_dir, 'traintab', 'model'))
    exp_instance = explainer.explain_tabular(train_df, ['true', 'false'], '8,-1.0,tuesday,word3',
                                             num_features=5)
    for i in range(2):
      label_data = exp_instance.as_list(label=i)
      # There should be 2 entries. One for categorical ("weekday") and one for numeric ("num")
      # label_data should look like:
      #    [
      #      ("weekday=tuesday", 0.02),
      #      ("num > 1.0", 0.03),
      #    ]
      self.assertEqual(2, len(label_data))
      keys = [x[0] for x in label_data]
      self.assertIn('weekday=tuesday', keys)
      keys.remove('weekday=tuesday')
      self.assertTrue('num' in keys[0])
