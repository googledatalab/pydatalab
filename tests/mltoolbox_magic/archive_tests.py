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

"""Test MLToolbox Magic's archive functions."""
from __future__ import absolute_import
from __future__ import print_function


import os
import shutil
import subprocess
import tempfile
import unittest

import google.datalab.contrib.mltoolbox._archive as _archive


class TestArchive(unittest.TestCase):
  """Tests for untar-ing files"""

  def setUp(self):
    self._root_folder = tempfile.mkdtemp()

    # Make two files to tar/compress
    self._src_folder = tempfile.mkdtemp(dir=self._root_folder)
    self._filename1 = 'file1.txt'
    self._filename2 = 'file2.txt'
    with open(os.path.join(self._src_folder, self._filename1), 'w') as f:
      f.write('This is file 1')
    with open(os.path.join(self._src_folder, self._filename2), 'w') as f:
      f.write('and this is file 2')

  def tearDown(self):
    shutil.rmtree(self._root_folder)

  def test_extract_archive_targz(self):
    """Tests extracting tar.gz files."""

    # Make a tar.gz file
    archive_path = os.path.join(self._root_folder, 'test.tar.gz')
    cmd = ['tar', '-czf', archive_path, '-C', self._src_folder, self._filename1, self._filename2]
    subprocess.check_call(cmd)

    # Undo it
    dest = os.path.join(self._root_folder, 'output')
    _archive.extract_archive(archive_path, dest)

    expected_file1 = os.path.join(dest, self._filename1)
    expected_file2 = os.path.join(dest, self._filename2)
    self.assertTrue(os.path.isfile(expected_file1))
    self.assertTrue(os.path.isfile(expected_file2))

    with open(expected_file2, 'r') as f:
      file_contents = f.read()
    self.assertTrue(file_contents, 'and this is file2')

  def test_extract_archive_tar(self):
    """Tests extracting tar.gz files."""

    # Make a tar.gz file
    archive_path = os.path.join(self._root_folder, 'test.tar')
    cmd = ['tar', '-cf', archive_path, '-C', self._src_folder, self._filename1, self._filename2]
    subprocess.check_call(cmd)

    # Undo it
    dest = os.path.join(self._root_folder, 'output')
    _archive.extract_archive(archive_path, dest)

    expected_file1 = os.path.join(dest, self._filename1)
    expected_file2 = os.path.join(dest, self._filename2)
    self.assertTrue(os.path.isfile(expected_file1))
    self.assertTrue(os.path.isfile(expected_file2))

    with open(expected_file2, 'r') as f:
      file_contents = f.read()
    self.assertTrue(file_contents, 'and this is file2')


if __name__ == '__main__':
    unittest.main()
