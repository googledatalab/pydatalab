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

  def _create_udf(self, name='test_udf', code='console.log("test");', return_type='integer',
                  params=None, language='js', imports=None):
    if params is None:
      params = {'test_param': 'integer'}
    if code is None:
      code = 'test code;'
    if imports is None:
      imports = ['gcs://mylib']
    return google.datalab.bigquery.UDF(name, code, return_type, params, language, imports)

  def test_building_udf(self):
    code = 'console.log("test");'
    imports = ['gcs://test_lib']
    udf = self._create_udf(code=code, imports=imports)

    expected_sql = '''\
CREATE TEMPORARY FUNCTION test_udf (test_param integer)
RETURNS integer
LANGUAGE js
AS """
console.log("test");
"""
OPTIONS (
library="gcs://test_lib"
);\
'''
    self.assertEqual(udf.name, 'test_udf')
    self.assertEqual(udf.code, code)
    self.assertEqual(udf.imports, imports)

    self.assertEqual(udf._language, 'js')
    self.assertEqual(udf._repr_sql_(), udf._expanded_sql())
    self.assertEqual(udf.__repr__(), 'BigQuery UDF - code:\n%s' % udf._code)
    self.assertEqual(udf._expanded_sql(), expected_sql)

  def test_udf_bad_return_type(self):
    with self.assertRaisesRegexp(TypeError, 'Argument return_type should be a string'):
      self._create_udf(return_type=['integer'])

  def test_udf_bad_params(self):
    with self.assertRaisesRegexp(TypeError, 'Argument params should be a dictionary'):
      self._create_udf(params=['param1', 'param2'])

  def test_udf_bad_imports(self):
    with self.assertRaisesRegexp(TypeError, 'Argument imports should be a list'):
      self._create_udf(imports='gcs://mylib')

  def test_udf_imports_non_js(self):
    with self.assertRaisesRegexp(Exception, 'Imports are available for Javascript'):
      self._create_udf(language='sql')

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
