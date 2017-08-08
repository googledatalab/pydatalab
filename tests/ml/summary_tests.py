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
from __future__ import unicode_literals
import unittest

import os
import pandas as pd
import shutil
import tensorflow as tf
import tempfile

from google.datalab.ml import Summary


class TestSummary(unittest.TestCase):
  """Tests google.datalab.ml.Summary class."""

  def setUp(self):
    self._test_dir = tempfile.mkdtemp()
    self._create_events()

  def tearDown(self):
    shutil.rmtree(self._test_dir)

  def _create_events(self):
    with tf.Session(graph=tf.Graph()) as sess:
      train_num = tf.placeholder(dtype=tf.float32, shape=[])
      eval_num1 = tf.multiply(train_num, 2)
      eval_num2 = tf.add(eval_num1, 10)
      train_summary = tf.summary.scalar('train_num', train_num)
      eval_summary1 = tf.summary.scalar('eval_num1', eval_num1)
      eval_summary2 = tf.summary.scalar('eval_num2', eval_num2)
      train_writer = tf.summary.FileWriter(os.path.join(self._test_dir, 'train'), sess.graph)
      eval_writer = tf.summary.FileWriter(os.path.join(self._test_dir, 'train', 'eval'), sess.graph)
      for i in range(10):
        t, evr1, evr2 = sess.run(
            [train_summary, eval_summary1, eval_summary2], feed_dict={train_num: i + 1})
        train_writer.add_summary(t, i)
        eval_writer.add_summary(t, i)
        eval_writer.add_summary(evr1, i)
        eval_writer.add_summary(evr2, i)
      train_writer.close()
      eval_writer.close()

  def test_list_events(self):
    """Tests list_events()."""

    train_dir = os.path.join(self._test_dir, 'train')
    eval_dir = os.path.join(self._test_dir, 'train', 'eval')
    summary = Summary(train_dir)
    events_dict = summary.list_events()
    expected_events_dict = {
      'train_num': {train_dir, eval_dir},
      'eval_num1': {eval_dir},
      'eval_num2': {eval_dir},
    }

    self.assertEqual(expected_events_dict, events_dict)

  def test_get_events(self):
    """Tests get_events()."""

    train_dir = os.path.join(self._test_dir, 'train')
    eval_dir = os.path.join(self._test_dir, 'train', 'eval')
    summary = Summary(train_dir)
    events_list = summary.get_events(['train_num', 'eval_num1'])

    self.assertEqual(set(events_list[0].keys()), {train_dir, eval_dir})
    self.assertEqual(set(events_list[1].keys()), {eval_dir})

    df = events_list[0][train_dir]
    self.assertEqual(list(range(0, 10)), df['step'].tolist())
    self.assertEqual(list(range(1, 11)), df['value'].tolist())
    self.assertIsInstance(df['time'][0], pd.tslib.Timestamp)

    df = events_list[1][eval_dir]
    self.assertEqual(list(range(0, 10)), df['step'].tolist())
    self.assertEqual(list(range(2, 22, 2)), df['value'].tolist())

  def test_plot_events(self):
    """Tests plot_events()."""

    train_dir = os.path.join(self._test_dir, 'train')
    summary = Summary(train_dir)
    # Call the function and make sure no exception.
    summary.plot('eval_num2')
    summary.plot(['train_num', 'eval_num2', 'eval_num2'])
