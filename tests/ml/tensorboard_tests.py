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


from google.datalab.ml import TensorBoard


class TestTensorboard(unittest.TestCase):
  def test_tensorboard(self):
    df = TensorBoard.list()
    if not df.empty:
      for pid in df['pid']:
        TensorBoard.stop(int(pid))

    TensorBoard.start('./a')
    TensorBoard.start('./b')
    df = TensorBoard.list()
    self.assertEqual(2, len(df))
    self.assertEqual(set(df['logdir']), {'./a', './b'})
    for pid in df['pid']:
      TensorBoard.stop(pid)

    # It seems on travis psutil.kill doesn't work. The following passes
    # on my workstation but not travis. Disable for now.
    # df = TensorBoard.list()
    # self.assertTrue(df.empty)
