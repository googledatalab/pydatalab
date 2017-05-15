# Copyright 2015 Google Inc. All rights reserved.
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
import google.datalab.bigquery


class TestCases(unittest.TestCase):

  def test_building_udf(self):
    code = 'console.log("test");'
    return_type = 'integer'
    params = {'test_param': 'integer'}
    language = 'js'
    imports = ['gcs://mylib']
    udf = google.datalab.bigquery.UDF('test_udf', code, return_type, params, language, imports)

    expected_sql = '''\
CREATE TEMPORARY FUNCTION test_udf (test_param integer)
RETURNS integer
LANGUAGE js
AS """
console.log("test");
"""
OPTIONS (
library="gcs://mylib"
);\
'''
    self.assertEqual(udf._expanded_sql(), expected_sql)

    # bad return type
    return_type = ['integer']
    with self.assertRaises(TypeError):
      google.datalab.bigquery.UDF('test_udf', code, return_type, params, language, imports)
    return_type = 'integer'

    # bad params
    params = ['param1', 'param2']
    with self.assertRaises(TypeError):
      google.datalab.bigquery.UDF('test_udf', code, return_type, params, language, imports)
    params = {'param1': 'integer'}

    # bad imports
    imports = 'gcs://mylib'
    with self.assertRaises(TypeError):
      google.datalab.bigquery.UDF('test_udf', code, return_type, params, language, imports)
    imports = ['gcs://mylib']

  def test_query_with_udf(self):
    code = 'console.log("test");'
    return_type = 'integer'
    params = {'test_param': 'integer'}
    language = 'js'
    imports = ''
    udf = google.datalab.bigquery.UDF('test_udf', code, return_type, params, language, imports)
    sql = 'SELECT test_udf(col) FROM mytable'
    expected_sql = '''\
CREATE TEMPORARY FUNCTION test_udf (test_param integer)
RETURNS integer
LANGUAGE js
AS """
console.log("test");
"""
OPTIONS (

);
SELECT test_udf(col) FROM mytable\
'''
    query = google.datalab.bigquery.Query(sql, udfs={'udf': udf})
    self.assertEquals(query.sql, expected_sql)

    # Alternate form of passing the udf using notebook environment
    query = google.datalab.bigquery.Query(sql, udfs=['udf'], env={'udf': udf})
    self.assertEquals(query.sql, expected_sql)
