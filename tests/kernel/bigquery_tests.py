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
import google.datalab.bigquery.commands
import google.datalab.utils.commands


class TestCases(unittest.TestCase):

  @mock.patch('google.datalab.utils.commands.notebook_environment')
  @mock.patch('google.datalab.Context.default')
  def test_udf_cell(self, mock_default_context, mock_notebook_environment):
    env = {}
    cell_body = \
"""
  // @param word STRING
  // @param corpus STRING
  // @returns INTEGER
  re = new RegExp(word, 'g');
  return corpus.match(re || []).length;
"""
    mock_default_context.return_value = TestCases._create_context()
    mock_notebook_environment.return_value = env
    google.datalab.bigquery.commands._bigquery._udf_cell({'name': 'count_occurrences',
                                                          'language': 'js'}, cell_body)
    udf = env['count_occurrences']
    self.assertIsNotNone(udf)
    self.assertEquals('count_occurrences', udf._name)
    self.assertEquals('js', udf._language)
    self.assertEquals('INTEGER', udf._return_type)
    self.assertEquals([('word', 'STRING'), ('corpus', 'STRING')], udf._params)
    self.assertEquals([], udf._imports)

  @mock.patch('google.datalab.utils.commands.notebook_environment')
  @mock.patch('google.datalab.Context.default')
  def test_query_cell(self, mock_default_context, mock_notebook_environment):
    env = {}
    mock_default_context.return_value = TestCases._create_context()
    mock_notebook_environment.return_value = env
    IPython.get_ipython().user_ns = env

    # test query creation
    q1_body = 'SELECT * FROM test_table'
    google.datalab.bigquery.commands._bigquery._query_cell({'name': 'q1', 'udfs': None,
                                                'datasources': None, 'subqueries': None,
                                                'parameters': None}, q1_body)
    q1 = env['q1']
    self.assertIsNotNone(q1)
    self.assertIsNone(q1._udfs)
    self.assertIsNone(q1._subqueries)
    self.assertEqual(q1_body, q1._sql)
    self.assertEqual(q1_body, q1.sql)

    # test subquery reference and expansion
    q2_body = 'SELECT * FROM q1'
    google.datalab.bigquery.commands._bigquery._query_cell({'name': 'q2', 'udfs': None,
                                                'datasources': None, 'subqueries': ['q1'],
                                                'parameters': None}, q2_body)
    q2 = env['q2']
    self.assertIsNotNone(q2)
    self.assertIsNone(q2._udfs)
    self.assertEqual(['q1'], q2._subqueries)
    expected_sql = 'WITH q1 AS (%s)\n%s' % (q1_body, q2_body)
    self.assertEqual(q2_body, q2._sql)
    self.assertEqual(expected_sql, q2.sql)

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = AccessTokenCredentials('test_token', 'test_ua')
    return google.datalab.Context(project_id, creds)

  def test_sample_cell(self):
    # TODO(gram): complete this test
    pass

  def test_get_schema(self):
    # TODO(gram): complete this test
    pass

  def test_get_table(self):
    # TODO(gram): complete this test
    pass

  def test_table_viewer(self):
    # TODO(gram): complete this test
    pass
