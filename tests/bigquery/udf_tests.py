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
from oauth2client.client import AccessTokenCredentials
import unittest

import google.datalab.bigquery
import google.datalab.context


class TestCases(unittest.TestCase):

  def test_sql_building(self):
    context = self._create_context()
    table = google.datalab.bigquery.Table('test:requestlogs.today', context=context)

    udf = self._create_udf()
    query = google.datalab.bigquery.Query('SELECT * FROM foo($t)', t=table, udfs=[udf], context=context)

    expected_js = '\nfoo=function(r,emit) { emit({output1: r.field2, output2: r.field1 }); };\n' +\
                  'bigquery.defineFunction(\'foo\', ["field1", "field2"], ' +\
                  '[{"name": "output1", "type": "integer"}, ' +\
                  '{"name": "output2", "type": "string"}], foo);'
    self.assertEqual(query.sql, 'SELECT * FROM (SELECT output1, output2 FROM foo([test:requestlogs.today]))')
    self.assertEqual(udf._code, expected_js)

  def test_udf_expansion(self):
    sql = 'SELECT * FROM udf(source)'
    udf = datalab.bigquery.UDF('inputs', [('foo', 'string'), ('bar', 'integer')], 'udf', 'code')
    context = TestCases._create_context()
    query = datalab.bigquery.Query(sql, udf=udf, context=context)
    self.assertEquals('SELECT * FROM (SELECT foo, bar FROM udf(source))', query.sql)

    # Alternate form
    query = datalab.bigquery.Query(sql, udfs=[udf], context=context)
    self.assertEquals('SELECT * FROM (SELECT foo, bar FROM udf(source))', query.sql)

  @staticmethod
  def _create_udf():
    inputs = [('field1', 'string'), ('field2', 'integer')]
    outputs = [('output1', 'integer'), ('output2', 'string')]
    impl = 'function(r,emit) { emit({output1: r.field2, output2: r.field1 }); }'
    udf = google.datalab.bigquery.UDF(inputs, outputs, 'foo', impl)
    return udf

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = AccessTokenCredentials('test_token', 'test_ua')
    return google.datalab.context.Context(project_id, creds)
