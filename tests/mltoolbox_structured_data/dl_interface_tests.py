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


import logging
import os
import sys

import shutil
import sys
import tempfile
import unittest

import mltoolbox._structured_data as core_sd
import google.datalab.ml as dlml


class TestAnalyze(unittest.TestCase):

  def test_datasets(self):

    with self.assertRaises(ValueError) as error:
      core_sd.analyze('some_dir', 'some_file.txt')
    self.assertIn('Only CsvDataSet is supported', str(error))

    with self.assertRaises(ValueError) as error:
      core_sd.analyze(
          'some_dir', 
          dlml.CsvDataSet(
              file_pattern=['file1.txt', 'file2.txt'],
              schema='col1:STRING,col2:INTEGER,col3:FLOAT'))
    self.assertIn('should be built with a file pattern', str(error))    
