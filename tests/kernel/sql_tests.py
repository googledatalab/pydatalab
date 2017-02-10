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
import imp
import mock
from oauth2client.client import AccessTokenCredentials
import unittest

# import Python so we can mock the parts we need to here.
import IPython
import IPython.core.magic


def noop_decorator(func):
  return func

IPython.core.magic.register_line_cell_magic = noop_decorator
IPython.core.magic.register_line_magic = noop_decorator
IPython.core.magic.register_cell_magic = noop_decorator
IPython.get_ipython = mock.Mock()

import google.datalab
import google.datalab.bigquery
import google.datalab.data
import google.datalab.data.commands
import google.datalab.utils.commands


class TestCases(unittest.TestCase):

  _SQL_MODULE_MAIN = google.datalab.data._utils._SQL_MODULE_MAIN
  _SQL_MODULE_LAST = google.datalab.data._utils._SQL_MODULE_LAST

  def test_split_cell(self):
    # TODO(gram): add tests for argument parser.
    m = imp.new_module('m')
    query = google.datalab.data.commands._sql._split_cell('', m)
    self.assertIsNone(query)
    self.assertNotIn(TestCases._SQL_MODULE_LAST, m.__dict__)
    self.assertNotIn(TestCases._SQL_MODULE_MAIN, m.__dict__)

    m = imp.new_module('m')
    query = google.datalab.data.commands._sql._split_cell('\n\n', m)
    self.assertIsNone(query)
    self.assertNotIn(TestCases._SQL_MODULE_LAST, m.__dict__)
    self.assertNotIn(TestCases._SQL_MODULE_MAIN, m.__dict__)

    m = imp.new_module('m')
    query = google.datalab.data.commands._sql._split_cell('# This is a comment\n\nSELECT 3 AS x', m)
    self.assertEquals(query, m.__dict__[TestCases._SQL_MODULE_MAIN])
    self.assertEquals(query, m.__dict__[TestCases._SQL_MODULE_LAST])
    self.assertEquals('SELECT 3 AS x', m.__dict__[TestCases._SQL_MODULE_MAIN].sql)
    self.assertEquals('SELECT 3 AS x', m.__dict__[TestCases._SQL_MODULE_LAST].sql)

    m = imp.new_module('m')
    query = google.datalab.data.commands._sql._split_cell(
        '# This is a comment\n\nfoo="bar"\nSELECT 3 AS x', m)
    self.assertEquals(query, m.__dict__[TestCases._SQL_MODULE_MAIN])
    self.assertEquals(query, m.__dict__[TestCases._SQL_MODULE_LAST])
    self.assertEquals('SELECT 3 AS x', m.__dict__[TestCases._SQL_MODULE_MAIN].sql)
    self.assertEquals('SELECT 3 AS x', m.__dict__[TestCases._SQL_MODULE_LAST].sql)

    sql_string_list = ['SELECT 3 AS x',
                       'WITH q1 as (SELECT "1")\nSELECT * FROM q1',
                       'INSERT DataSet.Table (Id, Description)\n' +
                           'VALUES(100,"TestDesc")',
                       'INSERT DataSet.Table (Id, Description)\n' +
                           'SELECT * FROM UNNEST([(200,"TestDesc2"),(300,"TestDesc3")])'
                       'INSERT DataSet.Table (Id, Description)\n' +
                           'WITH w as (SELECT ARRAY<STRUCT<Id int64, Description string>>\n' +
                           '[(400, "TestDesc4"),(500, "TestDesc5")] col)\n' +
                           'SELECT Id, Description FROM w, UNNEST(w.col)'
                       'INSERT DataSet.Table (Id, Description)\n' +
                           'VALUES (600,\n' +
                           '(SELECT Description FROM DataSet.Table WHERE Id = 400))',
                       'DELETE FROM DataSet.Table WHERE DESCRIPTION IS NULL'
                       'DELETE FROM DataSet.Table\n' +
                           'WHERE Id NOT IN (100, 200, 300)'
                       ]
    for i in range(0, len(sql_string_list)):
        m = imp.new_module('m')
        query = google.datalab.data.commands._sql._split_cell(sql_string_list[i], m)
        self.assertEquals(query, m.__dict__[TestCases._SQL_MODULE_MAIN])
        self.assertEquals(query, m.__dict__[TestCases._SQL_MODULE_LAST])
        self.assertEquals(sql_string_list[i], m.__dict__[TestCases._SQL_MODULE_MAIN].sql)
        self.assertEquals(sql_string_list[i], m.__dict__[TestCases._SQL_MODULE_LAST].sql)

    m = imp.new_module('m')
    query = google.datalab.data.commands._sql._split_cell('DEFINE QUERY q1\nSELECT 3 AS x', m)
    self.assertEquals(query, m.__dict__[TestCases._SQL_MODULE_LAST])
    self.assertEquals(query, m.__dict__[TestCases._SQL_MODULE_LAST])
    self.assertEquals('SELECT 3 AS x', m.q1.sql)
    self.assertNotIn(TestCases._SQL_MODULE_MAIN, m.__dict__)
    self.assertEquals('SELECT 3 AS x', m.__dict__[TestCases._SQL_MODULE_LAST].sql)

    m = imp.new_module('m')
    query = google.datalab.data.commands._sql._split_cell(
        'DEFINE QUERY q1\nSELECT 3 AS x\nSELECT * FROM $q1', m)
    self.assertEquals(query, m.__dict__[TestCases._SQL_MODULE_MAIN])
    self.assertEquals(query, m.__dict__[TestCases._SQL_MODULE_LAST])
    self.assertEquals('SELECT 3 AS x', m.q1.sql)
    self.assertEquals('SELECT * FROM $q1', m.__dict__[TestCases._SQL_MODULE_MAIN].sql)
    self.assertEquals('SELECT * FROM $q1', m.__dict__[TestCases._SQL_MODULE_LAST].sql)

  @mock.patch('google.datalab.Context.default')
  def test_arguments(self, mock_default_context):
    mock_default_context.return_value = TestCases._create_context()
    m = imp.new_module('m')
    query = google.datalab.data.commands._sql._split_cell("""
words = ('thus', 'forsooth')
limit = 10

SELECT * FROM [publicdata:samples.shakespeare]
WHERE word IN $words
LIMIT $limit
""", m)
    sql = google.datalab.bigquery.Query(query, env={}).sql
    self.assertEquals('SELECT * FROM [publicdata:samples.shakespeare]\n' +
                      'WHERE word IN ("thus", "forsooth")\nLIMIT 10', sql)
    # As above but with overrides, using list
    sql = google.datalab.bigquery.Query(query, env={'words': 'eyeball', 'limit': 5}).sql
    self.assertEquals('SELECT * FROM [publicdata:samples.shakespeare]\n' +
                      'WHERE word IN "eyeball"\nLIMIT 5', sql)
    # As above but with overrides, using tuple and env dict
    sql = google.datalab.bigquery.Query(query, env={'limit': 3, 'words': ('thus',)}).sql
    self.assertEquals('SELECT * FROM [publicdata:samples.shakespeare]\n' +
                      'WHERE word IN ("thus")\nLIMIT 3', sql)
    # As above but with list argument
    m = imp.new_module('m')
    query = google.datalab.data.commands._sql._split_cell("""
words = ['thus', 'forsooth']
limit = 10

SELECT * FROM [publicdata:samples.shakespeare]
WHERE word IN $words
LIMIT $limit
""", m)
    sql = google.datalab.bigquery.Query(query, env={}).sql
    self.assertEquals('SELECT * FROM [publicdata:samples.shakespeare]\n' +
                      'WHERE word IN ("thus", "forsooth")\nLIMIT 10', sql)
    # As above but with overrides, using list
    sql = google.datalab.bigquery.Query(query, env={'limit': 2, 'words': ['forsooth']}).sql
    self.assertEquals('SELECT * FROM [publicdata:samples.shakespeare]\n' +
                      'WHERE word IN ("forsooth")\nLIMIT 2', sql)
    # As above but with overrides, using tuple
    sql = google.datalab.bigquery.Query(query, env={'words': 'eyeball'}).sql
    self.assertEquals('SELECT * FROM [publicdata:samples.shakespeare]\n' +
                      'WHERE word IN "eyeball"\nLIMIT 10', sql)
    # TODO(gram): add some tests for source and datestring variables

  def test_date(self):
    # TODO(gram): complete this test
    pass

  def test_sql_cell(self):
    # TODO(gram): complete this test
    pass

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = AccessTokenCredentials('test_token', 'test_ua')
    return google.datalab.Context(project_id, creds)
