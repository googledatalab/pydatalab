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
"""Test the datalab interface functions in _package.py
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import six
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../..')))

import inspect  # noqa: E402
import mltoolbox._structured_data._package as core_sd  # noqa: E402
import mltoolbox.classification.linear as classlin  # noqa: E402
import mltoolbox.classification.dnn as classdnn  # noqa: E402
import mltoolbox.regression.linear as reglin  # noqa: E402
import mltoolbox.regression.dnn as regdnn  # noqa: E402
import google.datalab.ml as dlml  # noqa: E402
import unittest  # noqa: E402


@unittest.skipIf(not six.PY2, 'Python 2 is required')
class TestAnalyze(unittest.TestCase):

  def test_not_csvdataset(self):
    """Test csvdataset is used"""
    # not a CsvDataSet
    job = core_sd.analyze_async('some_dir', 'some_file.txt').wait()
    self.assertIn('Only CsvDataSet is supported', job.fatal_error.message)

  def test_csvdataset_one_file(self):
    """Test CsvDataSet has only one file/pattern"""
    # TODO(brandondutra) remove this restriction
    job = core_sd.analyze_async(
        'some_dir',
        dlml.CsvDataSet(
            file_pattern=['file1.txt', 'file2.txt'],
            schema='col1:STRING,col2:INTEGER,col3:FLOAT')).wait()
    self.assertIn('should be built with a file pattern',
                  job.fatal_error.message)

  def test_projectid(self):
    """Test passing project id but cloud is false"""
    job = core_sd.analyze_async(
        'some_dir',
        dlml.CsvDataSet(
            file_pattern=['file1.txt'],
            schema='col1:STRING,col2:INTEGER,col3:FLOAT'),
        project_id='project_id').wait()
    self.assertIn('project_id only needed if cloud is True',
                  job.fatal_error.message)

  def test_cloud_with_local_output_folder(self):
    job = core_sd.analyze_async(
        'some_dir',
        dlml.CsvDataSet(
            file_pattern=['gs://file1.txt'],
            schema='col1:STRING,col2:INTEGER,col3:FLOAT'),
        project_id='project_id',
        cloud=True).wait()
    self.assertIn('File some_dir is not a gcs path', job.fatal_error.message)

  def test_cloud_but_local_files(self):
    job = core_sd.analyze_async(
        'gs://some_dir',
        dlml.CsvDataSet(
            file_pattern=['file1.txt'],
            schema='col1:STRING,col2:INTEGER,col3:FLOAT'),
        project_id='project_id',
        cloud=True).wait()
    self.assertIn('File file1.txt is not a gcs path', job.fatal_error.message)

  def test_unsupported_schema(self):
    """Test supported schema values.

    Note that not all valid BQ schema values are valid/used in the structured
    data package
    """

    unsupported_col_types = ['bytes', 'boolean', 'timestamp', 'date', 'time',
                             'datetime', 'record']
    for col_type in unsupported_col_types:
      schema = 'col_name:%s' % col_type

      job = core_sd.analyze_async(
        'some_dir',
        dlml.CsvDataSet(
            file_pattern=['file1.txt'],
            schema=schema),
        cloud=False).wait()
      self.assertIn('Schema contains an unsupported type %s.' % col_type,
                    job.fatal_error.message)

      job = core_sd.analyze_async(
        'gs://some_dir',
        dlml.CsvDataSet(
            file_pattern=['gs://file1.txt'],
            schema=schema),
        cloud=True,
        project_id='junk_project_id').wait()
      self.assertIn('Schema contains an unsupported type %s.' % col_type,
                    job.fatal_error.message)


@unittest.skipIf(not six.PY2, 'Python 2 is required')
class TestFunctionSignature(unittest.TestCase):

  def _argspec(self, fn_obj):
    if six.PY2:
      return inspect.getargspec(fn_obj)
    else:
      return inspect.getfullargspec(fn_obj)

  def test_same_analysis(self):
    """Test that there is only one analyze function"""
    self.assertIs(core_sd.analyze, classlin.analyze)
    self.assertIs(core_sd.analyze, classdnn.analyze)
    self.assertIs(core_sd.analyze, reglin.analyze)
    self.assertIs(core_sd.analyze, regdnn.analyze)

  def test_same_analysis_async(self):
    """Test that there is only one analyze_async function"""
    self.assertIs(core_sd.analyze_async, classlin.analyze_async)
    self.assertIs(core_sd.analyze_async, classdnn.analyze_async)
    self.assertIs(core_sd.analyze_async, reglin.analyze_async)
    self.assertIs(core_sd.analyze_async, regdnn.analyze_async)

  def test_analysis_argspec(self):
    """Test all analyze functions have the same parameters"""

    self.assertEqual(self._argspec(core_sd.analyze),
                     self._argspec(core_sd.analyze_async))
    self.assertEqual(self._argspec(core_sd.analyze),
                     self._argspec(core_sd._analyze))
