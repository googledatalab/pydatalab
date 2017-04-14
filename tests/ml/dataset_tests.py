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

import google.datalab
from google.datalab.ml import CsvDataSet


class TestCases(unittest.TestCase):

  def test_create_dataset(self):
    json_schema = TestCases._create_json_schema()

    # CsvDataSet can take a json schema, a Schema object, or a string
    ds = CsvDataSet(file_pattern='some/file', schema=json_schema)
    self.assertEqual(json_schema, ds.schema)

    schema_obj = google.datalab.bigquery.Schema(json_schema)
    ds = CsvDataSet(file_pattern='some/file', schema=schema_obj)
    self.assertEqual(json_schema, ds.schema)

    schema_str = 'field1: STRING, field2: INTEGER'
    ds = CsvDataSet(file_pattern='some/file', schema=schema_str)
    self.assertEqual(json_schema, ds.schema)

  @staticmethod
  def _create_json_schema():
    return [{'name': 'field1', 'type': 'STRING'},
            {'name': 'field2', 'type': 'INTEGER'}]
