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

import google.datalab
import google.datalab.storage
import google.datalab.utils


class TestCases(unittest.TestCase):

  @mock.patch('google.datalab.storage._api.Api.objects_list')
  @mock.patch('google.datalab.storage._api.Api.objects_get')
  def test_object_existence(self, mock_api_objects_get, mock_api_objects_list):
    mock_api_objects_list.return_value = TestCases._create_enumeration_single_result()
    mock_api_objects_get.return_value = TestCases._create_objects_get_result()

    b = TestCases._create_bucket()
    self.assertTrue(b.objects().contains('test_object1'))

    mock_api_objects_get.side_effect = google.datalab.utils.RequestException(404, 'failed')
    self.assertFalse('test_object2' in list(b.objects()))

  @mock.patch('google.datalab.storage._api.Api.objects_get')
  def test_object_metadata(self, mock_api_objects):
    mock_api_objects.return_value = TestCases._create_objects_get_result()

    b = TestCases._create_bucket()
    i = b.object('test_object1')
    m = i.metadata

    self.assertEqual(m.name, 'test_object1')
    self.assertEqual(m.content_type, 'text/plain')

  @mock.patch('google.datalab.storage._api.Api.objects_list')
  def test_enumerate_objects_empty(self, mock_api_objects):
    mock_api_objects.return_value = TestCases._create_enumeration_empty_result()

    b = self._create_bucket()
    objects = list(b.objects())

    self.assertEqual(len(objects), 0)

  @mock.patch('google.datalab.storage._api.Api.objects_list')
  def test_enumerate_objects_single(self, mock_api_objects):
    mock_api_objects.return_value = TestCases._create_enumeration_single_result()

    b = TestCases._create_bucket()
    objects = list(b.objects())

    self.assertEqual(len(objects), 1)
    self.assertEqual(objects[0].key, 'test_object1')

  @mock.patch('google.datalab.storage._api.Api.objects_list')
  def test_enumerate_objects_multi_page(self, mock_api_objects):
    mock_api_objects.side_effect = [
      TestCases._create_enumeration_multipage_result1(),
      TestCases._create_enumeration_multipage_result2()
    ]

    b = TestCases._create_bucket()
    objects = list(b.objects())

    self.assertEqual(len(objects), 2)
    self.assertEqual(objects[0].key, 'test_object1')
    self.assertEqual(objects[1].key, 'test_object2')

  @mock.patch('google.datalab.storage._api.Api.objects_list')
  def test_object_delete_with_wait(self, mock_objects_list):
    stable_object_name = 'testobject'
    object_to_delete = 'temporaryobject'
    mock_objects_list.side_effect = [
        {'items': [{'name': stable_object_name}], 'nextPageToken': 'yes'},
        {'items': [{'name': object_to_delete}]},
        {'items': [{'name': stable_object_name}]},
    ]

    b = TestCases._create_bucket()
    o = b.object(object_to_delete)
    o._info = {'name': object_to_delete}

    with mock.patch.object(google.datalab.storage._api.Api, 'objects_delete',
                           autospec=True) as mock_objects_delete:
      o.delete(wait_for_deletion=False)
    self.assertEqual(1, mock_objects_delete.call_count)
    # storage.objects.list shouldn't have been called with
    # wait_for_deletion=False.
    self.assertEqual(0, mock_objects_list.call_count)

    with mock.patch.object(google.datalab.storage._api.Api, 'objects_delete',
                           autospec=True) as mock_objects_delete:
      o.delete()
    self.assertEqual(1, mock_objects_delete.call_count)
    # storage.objects.list should have been called three times with
    # wait_for_deletion=True:
    #  * twice on the first run, paging through all results, with the object
    #    still present in the bucket, and
    #  * once on a second run, now with no object present in the list.
    self.assertEqual(3, mock_objects_list.call_count)

  @staticmethod
  def _create_bucket(name='test_bucket'):
    return google.datalab.storage.Bucket(name, context=TestCases._create_context())

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = AccessTokenCredentials('test_token', 'test_ua')
    return google.datalab.Context(project_id, creds)

  @staticmethod
  def _create_objects_get_result():
    return {'name': 'test_object1', 'contentType': 'text/plain'}

  @staticmethod
  def _create_enumeration_empty_result():
    return {}

  @staticmethod
  def _create_enumeration_single_result():
    return {
      'items': [
        {'name': 'test_object1'}
      ]
    }

  @staticmethod
  def _create_enumeration_multipage_result1():
    return {
      'items': [
        {'name': 'test_object1'}
      ],
      'nextPageToken': 'test_token'
    }

  @staticmethod
  def _create_enumeration_multipage_result2():
    return {
      'items': [
        {'name': 'test_object2'}
      ]
    }
