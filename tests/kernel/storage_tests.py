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
import google.datalab.storage
import google.datalab.storage.commands


class TestCases(unittest.TestCase):

  @mock.patch('google.datalab.storage._object.Object.exists', autospec=True)
  @mock.patch('google.datalab.storage._bucket.Bucket.objects', autospec=True)
  @mock.patch('google.datalab.storage._api.Api.objects_get', autospec=True)
  @mock.patch('google.datalab.Context.default')
  def test_expand_list(self, mock_context_default, mock_api_objects_get, mock_bucket_objects,
                       mock_object_exists):
    context = TestCases._create_context()
    mock_context_default.return_value = context

    # Mock API for testing for object existence. Fail if called with name that includes wild char.
    def object_exists_side_effect(*args, **kwargs):
      return args[0].key.find('*') < 0

    mock_object_exists.side_effect = object_exists_side_effect

    # Mock API for getting objects in a bucket.
    mock_bucket_objects.side_effect = TestCases._mock_bucket_objects_return(context)

    # Mock API for getting object metadata.
    mock_api_objects_get.side_effect = TestCases._mock_api_objects_get()

    objects = google.datalab.storage.commands._storage._expand_list(None)
    self.assertEqual([], objects)

    objects = google.datalab.storage.commands._storage._expand_list([])
    self.assertEqual([], objects)

    objects = google.datalab.storage.commands._storage._expand_list('gs://bar/o*')
    self.assertEqual(['gs://bar/object1', 'gs://bar/object3'], objects)

    objects = google.datalab.storage.commands._storage._expand_list(['gs://foo', 'gs://bar'])
    self.assertEqual(['gs://foo', 'gs://bar'], objects)

    objects = google.datalab.storage.commands._storage._expand_list(['gs://foo/*', 'gs://bar'])
    self.assertEqual(['gs://foo/object1', 'gs://foo/object2', 'gs://foo/object3', 'gs://bar'], objects)

    objects = google.datalab.storage.commands._storage._expand_list(['gs://bar/o*'])
    self.assertEqual(['gs://bar/object1', 'gs://bar/object3'], objects)

    objects = google.datalab.storage.commands._storage._expand_list(['gs://bar/i*'])
    # Note - if no match we return the pattern.
    self.assertEqual(['gs://bar/i*'], objects)

    objects = google.datalab.storage.commands._storage._expand_list(['gs://baz'])
    self.assertEqual(['gs://baz'], objects)

    objects = google.datalab.storage.commands._storage._expand_list(['gs://baz/*'])
    self.assertEqual(['gs://baz/*'], objects)

    objects = google.datalab.storage.commands._storage._expand_list(['gs://foo/o*3'])
    self.assertEqual(['gs://foo/object3'], objects)

  @mock.patch('google.datalab.storage._object.Object.copy_to', autospec=True)
  @mock.patch('google.datalab.storage._bucket.Bucket.objects', autospec=True)
  @mock.patch('google.datalab.storage._api.Api.objects_get', autospec=True)
  @mock.patch('google.datalab.Context.default')
  def test_storage_copy(self, mock_context_default, mock_api_objects_get, mock_bucket_objects,
                        mock_storage_object_copy_to):
    context = TestCases._create_context()
    mock_context_default.return_value = context
    # Mock API for getting objects in a bucket.
    mock_bucket_objects.side_effect = TestCases._mock_bucket_objects_return(context)
    # Mock API for getting object metadata.
    mock_api_objects_get.side_effect = TestCases._mock_api_objects_get()

    google.datalab.storage.commands._storage._storage_copy({
      'source': ['gs://foo/object1'],
      'destination': 'gs://foo/bar1'
    }, None)

    mock_storage_object_copy_to.assert_called_with(mock.ANY, 'bar1', bucket='foo')
    self.assertEquals('object1', mock_storage_object_copy_to.call_args[0][0].key)
    self.assertEquals('foo', mock_storage_object_copy_to.call_args[0][0]._bucket)

    with self.assertRaises(Exception) as error:
      google.datalab.storage.commands._storage._storage_copy({
        'source': ['gs://foo/object*'],
        'destination': 'gs://foo/bar1'
      }, None)
    self.assertEqual('More than one source but target gs://foo/bar1 is not a bucket',
                     str(error.exception))

  @mock.patch('google.datalab.storage.commands._storage._storage_copy', autospec=True)
  def test_storage_copy_magic(self, mock_storage_copy):
    google.datalab.storage.commands._storage.storage('copy --source gs://foo/object1 --destination gs://foo/bar1')
    mock_storage_copy.assert_called_with({
        'source': ['gs://foo/object1'],
        'destination': 'gs://foo/bar1',
        'func': google.datalab.storage.commands._storage._storage_copy
      }, None)

  @mock.patch('google.datalab.storage._api.Api.buckets_insert', autospec=True)
  @mock.patch('google.datalab.Context.default')
  def test_storage_create(self, mock_context_default, mock_api_buckets_insert):
    context = TestCases._create_context()
    mock_context_default.return_value = context

    errs = google.datalab.storage.commands._storage._storage_create({
      'project': 'test',
      'bucket': [
        'gs://baz'
      ]
    }, None)
    self.assertEqual(None, errs)
    mock_api_buckets_insert.assert_called_with(mock.ANY, 'baz', project_id='test')

    with self.assertRaises(Exception) as error:
      google.datalab.storage.commands._storage._storage_create({
        'project': 'test',
        'bucket': [
          'gs://foo/bar'
        ]
      }, None)
    self.assertEqual("Couldn't create gs://foo/bar: Invalid bucket name gs://foo/bar",
                     str(error.exception))

  @mock.patch('google.datalab.storage._api.Api.buckets_get', autospec=True)
  @mock.patch('google.datalab.storage._api.Api.objects_get', autospec=True)
  @mock.patch('google.datalab.storage._bucket.Bucket.objects', autospec=True)
  @mock.patch('google.datalab.storage._api.Api.objects_delete', autospec=True)
  @mock.patch('google.datalab.storage._api.Api.buckets_delete', autospec=True)
  @mock.patch('google.datalab.Context.default')
  def test_storage_delete(self, mock_context_default, mock_api_bucket_delete,
                          mock_api_objects_delete, mock_bucket_objects, mock_api_objects_get,
                          mock_api_buckets_get):
    context = TestCases._create_context()
    mock_context_default.return_value = context
    # Mock API for getting objects in a bucket.
    mock_bucket_objects.side_effect = TestCases._mock_bucket_objects_return(context)
    # Mock API for getting object metadata.
    mock_api_objects_get.side_effect = TestCases._mock_api_objects_get()
    mock_api_buckets_get.side_effect = TestCases._mock_api_buckets_get()

    with self.assertRaises(Exception) as error:
      google.datalab.storage.commands._storage._storage_delete({
        'bucket': [
          'gs://bar',
          'gs://baz'
        ],
        'object': [
          'gs://foo/object1',
          'gs://baz/object1',
        ]
      }, None)
    self.assertEqual('gs://baz does not exist\ngs://baz/object1 does not exist',
                     str(error.exception))
    mock_api_bucket_delete.assert_called_with(mock.ANY, 'bar')
    mock_api_objects_delete.assert_called_with(mock.ANY, 'foo', 'object1')

  @mock.patch('google.datalab.Context.default')
  def test_storage_view(self, mock_context_default):
    context = TestCases._create_context()
    mock_context_default.return_value = context
    # TODO(gram): complete this test

  @mock.patch('google.datalab.Context.default')
  def test_storage_write(self, mock_context_default):
    context = TestCases._create_context()
    mock_context_default.return_value = context
    # TODO(gram): complete this test

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = AccessTokenCredentials('test_token', 'test_ua')
    return google.datalab.Context(project_id, creds)

  @staticmethod
  def _mock_bucket_objects_return(context):
    # Mock API for getting objects in a bucket.
    def bucket_objects_side_effect(*args, **kwargs):
      bucket = args[0].name  # self
      if bucket == 'foo':
        return [
          google.datalab.storage._object.Object(bucket, 'object1', context=context),
          google.datalab.storage._object.Object(bucket, 'object2', context=context),
          google.datalab.storage._object.Object(bucket, 'object3', context=context),
        ]
      elif bucket == 'bar':
        return [
          google.datalab.storage._object.Object(bucket, 'object1', context=context),
          google.datalab.storage._object.Object(bucket, 'object3', context=context),
        ]
      else:
        return []
    return bucket_objects_side_effect

  @staticmethod
  def _mock_api_objects_get():
    # Mock API for getting object metadata.
    def api_objects_get_side_effect(*args, **kwargs):
      if args[1].find('baz') >= 0:
        return None
      key = args[2]
      if key.find('*') >= 0:
        return None
      return {'name': key}
    return api_objects_get_side_effect

  @staticmethod
  def _mock_api_buckets_get():
    # Mock API for getting bucket metadata.
    def api_buckets_get_side_effect(*args, **kwargs):
      key = args[1]
      if key.find('*') >= 0 or key.find('baz') >= 0:
        return None
      return {'name': key}
    return api_buckets_get_side_effect
