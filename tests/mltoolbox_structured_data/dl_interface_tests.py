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
from __future__ import print_function

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                '../../solutionbox/structured_data/')))

import mltoolbox._structured_data._package as core_sd  # noqa: E402
import google.datalab.ml as dlml  # noqa: E402


class TestAnalyze(unittest.TestCase):

  def test_bad_parameters(self):
    """Test incorrect parameters.

    Test for things like using local files when requesting a cloud run.
    """
    # not a CsvDataSet
    job = core_sd.analyze_async('some_dir', 'some_file.txt').wait()
    self.assertIn('Only CsvDataSet is supported', job.fatal_error.message)

    # CsvDataSet does not have one file/pattern
    # TODO(brandondutra) remove this restriction
    job = core_sd.analyze_async(
        'some_dir',
        dlml.CsvDataSet(
            file_pattern=['file1.txt', 'file2.txt'],
            schema='col1:STRING,col2:INTEGER,col3:FLOAT')).wait()
    self.assertIn('should be built with a file pattern',
                  job.fatal_error.message)

    # Passing project id but cloud is false
    job = core_sd.analyze_async(
        'some_dir',
        dlml.CsvDataSet(
            file_pattern=['file1.txt'],
            schema='col1:STRING,col2:INTEGER,col3:FLOAT'),
        project_id='project_id').wait()
    self.assertIn('project_id only needed if cloud is True',
                  job.fatal_error.message)

    # Use cloud but local output folder
    job = core_sd.analyze_async(
        'some_dir',
        dlml.CsvDataSet(
            file_pattern=['gs://file1.txt'],
            schema='col1:STRING,col2:INTEGER,col3:FLOAT'),
        project_id='project_id',
        cloud=True).wait()
    self.assertIn('File some_dir is not a gcs path', job.fatal_error.message)

    # Use cloud, but local files
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
        cloud=True).wait()
      self.assertIn('Schema contains an unsupported type %s.' % col_type,
                    job.fatal_error.message)

  def test_numerical_data_with_categorical_schema(self):
    pass

  def test_categorical_data_with_numerical_schema(self):
    pass

  def test_extra_columns_local(self):
    pass
    # test more cols than schema
    # test more schema than cols?

  def test_function_signatures(self):
    pass
    # test blocking, async, and _analy all have same parameters.
