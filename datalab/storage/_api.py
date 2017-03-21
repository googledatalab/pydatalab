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

"""Implements Storage HTTP API wrapper."""
from __future__ import absolute_import
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()  # noqa
from builtins import object

import urllib.request
import urllib.parse
import urllib.error
import datalab.context
import datalab.utils


class Api(object):
  """A helper class to issue Storage HTTP requests."""

  # TODO(nikhilko): Use named placeholders in these string templates.
  _ENDPOINT = 'https://www.googleapis.com/storage/v1'
  _DOWNLOAD_ENDPOINT = 'https://www.googleapis.com/download/storage/v1'
  _UPLOAD_ENDPOINT = 'https://www.googleapis.com/upload/storage/v1'
  _BUCKET_PATH = '/b/%s'
  _OBJECT_PATH = '/b/%s/o/%s'
  _OBJECT_COPY_PATH = '/b/%s/o/%s/copyTo/b/%s/o/%s'

  _MAX_RESULTS = 100

  def __init__(self, context):
    """Initializes the Storage helper with context information.

    Args:
      context: a Context object providing project_id and credentials.
    """
    self._credentials = context.credentials
    self._project_id = context.project_id

  @property
  def project_id(self):
    """The project_id associated with this API client."""
    return self._project_id

  def buckets_insert(self, bucket, project_id=None):
    """Issues a request to create a new bucket.

    Args:
      bucket: the name of the bucket.
      project_id: the project to use when inserting the bucket.
    Returns:
      A parsed bucket information dictionary.
    Raises:
      Exception if there is an error performing the operation.
    """
    args = {'project': project_id if project_id else self._project_id}
    data = {'name': bucket}

    url = Api._ENDPOINT + (Api._BUCKET_PATH % '')
    return datalab.utils.Http.request(url, args=args, data=data, credentials=self._credentials)

  def buckets_delete(self, bucket):
    """Issues a request to delete a bucket.

    Args:
      bucket: the name of the bucket.
    Raises:
      Exception if there is an error performing the operation.
    """
    url = Api._ENDPOINT + (Api._BUCKET_PATH % bucket)
    datalab.utils.Http.request(url, method='DELETE', credentials=self._credentials,
                               raw_response=True)

  def buckets_get(self, bucket, projection='noAcl'):
    """Issues a request to retrieve information about a bucket.

    Args:
      bucket: the name of the bucket.
      projection: the projection of the bucket information to retrieve.
    Returns:
      A parsed bucket information dictionary.
    Raises:
      Exception if there is an error performing the operation.
    """
    args = {'projection': projection}
    url = Api._ENDPOINT + (Api._BUCKET_PATH % bucket)
    return datalab.utils.Http.request(url, credentials=self._credentials, args=args)

  def buckets_list(self, projection='noAcl', max_results=0, page_token=None, project_id=None):
    """Issues a request to retrieve the list of buckets.

    Args:
      projection: the projection of the bucket information to retrieve.
      max_results: an optional maximum number of objects to retrieve.
      page_token: an optional token to continue the retrieval.
      project_id: the project whose buckets should be listed.
    Returns:
      A parsed list of bucket information dictionaries.
    Raises:
      Exception if there is an error performing the operation.
    """
    if max_results == 0:
      max_results = Api._MAX_RESULTS

    args = {'project': project_id if project_id else self._project_id, 'maxResults': max_results}
    if projection is not None:
      args['projection'] = projection
    if page_token is not None:
      args['pageToken'] = page_token

    url = Api._ENDPOINT + (Api._BUCKET_PATH % '')
    return datalab.utils.Http.request(url, args=args, credentials=self._credentials)

  def object_download(self, bucket, key, start_offset=0, byte_count=None):
    """Reads the contents of an object as text.

    Args:
      bucket: the name of the bucket containing the object.
      key: the key of the object to be read.
      start_offset: the start offset of bytes to read.
      byte_count: the number of bytes to read. If None, it reads to the end.
    Returns:
      The text content within the object.
    Raises:
      Exception if the object could not be read from.
    """
    args = {'alt': 'media'}
    headers = {}
    if start_offset > 0 or byte_count is not None:
      header = 'bytes=%d-' % start_offset
      if byte_count is not None:
        header += '%d' % byte_count
      headers['Range'] = header
    url = Api._DOWNLOAD_ENDPOINT + (Api._OBJECT_PATH % (bucket, Api._escape_key(key)))
    return datalab.utils.Http.request(url, args=args, headers=headers,
                                      credentials=self._credentials, raw_response=True)

  def object_upload(self, bucket, key, content, content_type):
    """Writes text content to the object.

    Args:
      bucket: the name of the bucket containing the object.
      key: the key of the object to be written.
      content: the text content to be written.
      content_type: the type of text content.
    Raises:
      Exception if the object could not be written to.
    """
    args = {'uploadType': 'media', 'name': key}
    headers = {'Content-Type': content_type}

    url = Api._UPLOAD_ENDPOINT + (Api._OBJECT_PATH % (bucket, ''))
    return datalab.utils.Http.request(url, args=args, data=content, headers=headers,
                                      credentials=self._credentials, raw_response=True)

  def objects_copy(self, source_bucket, source_key, target_bucket, target_key):
    """Updates the metadata associated with an object.

    Args:
      source_bucket: the name of the bucket containing the source object.
      source_key: the key of the source object being copied.
      target_bucket: the name of the bucket that will contain the copied object.
      target_key: the key of the copied object.
    Returns:
      A parsed object information dictionary.
    Raises:
      Exception if there is an error performing the operation.
    """
    url = Api._ENDPOINT + (Api._OBJECT_COPY_PATH % (source_bucket, Api._escape_key(source_key),
                                                    target_bucket, Api._escape_key(target_key)))
    return datalab.utils.Http.request(url, method='POST', credentials=self._credentials)

  def objects_delete(self, bucket, key):
    """Deletes the specified object.

    Args:
      bucket: the name of the bucket.
      key: the key of the object within the bucket.
    Raises:
      Exception if there is an error performing the operation.
    """
    url = Api._ENDPOINT + (Api._OBJECT_PATH % (bucket, Api._escape_key(key)))
    datalab.utils.Http.request(url, method='DELETE', credentials=self._credentials,
                               raw_response=True)

  def objects_get(self, bucket, key, projection='noAcl'):
    """Issues a request to retrieve information about an object.

    Args:
      bucket: the name of the bucket.
      key: the key of the object within the bucket.
      projection: the projection of the object to retrieve.
    Returns:
      A parsed object information dictionary.
    Raises:
      Exception if there is an error performing the operation.
    """
    args = {}
    if projection is not None:
      args['projection'] = projection

    url = Api._ENDPOINT + (Api._OBJECT_PATH % (bucket, Api._escape_key(key)))
    return datalab.utils.Http.request(url, args=args, credentials=self._credentials)

  def objects_list(self, bucket, prefix=None, delimiter=None, projection='noAcl', versions=False,
                   max_results=0, page_token=None):
    """Issues a request to retrieve information about an object.

    Args:
      bucket: the name of the bucket.
      prefix: an optional key prefix.
      delimiter: an optional key delimiter.
      projection: the projection of the objects to retrieve.
      versions: whether to list each version of a file as a distinct object.
      max_results: an optional maximum number of objects to retrieve.
      page_token: an optional token to continue the retrieval.
    Returns:
      A parsed list of object information dictionaries.
    Raises:
      Exception if there is an error performing the operation.
    """
    if max_results == 0:
      max_results = Api._MAX_RESULTS

    args = {'maxResults': max_results}
    if prefix is not None:
      args['prefix'] = prefix
    if delimiter is not None:
      args['delimiter'] = delimiter
    if projection is not None:
      args['projection'] = projection
    if versions:
      args['versions'] = 'true'
    if page_token is not None:
      args['pageToken'] = page_token

    url = Api._ENDPOINT + (Api._OBJECT_PATH % (bucket, ''))
    return datalab.utils.Http.request(url, args=args, credentials=self._credentials)

  def objects_patch(self, bucket, key, info):
    """Updates the metadata associated with an object.

    Args:
      bucket: the name of the bucket containing the object.
      key: the key of the object being updated.
      info: the metadata to update.
    Returns:
      A parsed object information dictionary.
    Raises:
      Exception if there is an error performing the operation.
    """
    url = Api._ENDPOINT + (Api._OBJECT_PATH % (bucket, Api._escape_key(key)))
    return datalab.utils.Http.request(url, method='PATCH', data=info, credentials=self._credentials)

  @staticmethod
  def _escape_key(key):
    # Disable the behavior to leave '/' alone by explicitly specifying the safe parameter.
    return urllib.parse.quote(key, safe='')

  @staticmethod
  def verify_permitted_to_read(gs_path):
    """Check if the user has permissions to read from the given path.

    Args:
      gs_path: the GCS path to check if user is permitted to read.
    Raises:
      Exception if user has no permissions to read.
    """
    # TODO(qimingj): Storage APIs need to be modified to allow absence of project
    #                or credential on Items. When that happens we can move the function
    #                to Items class.
    from . import _bucket
    bucket, prefix = _bucket.parse_name(gs_path)
    credentials = None
    if datalab.context.Context.is_signed_in():
      credentials = datalab.context._utils.get_credentials()
    args = {
        'maxResults': Api._MAX_RESULTS,
        'projection': 'noAcl'
    }
    if prefix is not None:
      args['prefix'] = prefix
    url = Api._ENDPOINT + (Api._OBJECT_PATH % (bucket, ''))
    try:
      datalab.utils.Http.request(url, args=args, credentials=credentials)
    except datalab.utils.RequestException as e:
      if e.status == 401:
        raise Exception('Not permitted to read from specified path. '
                        'Please sign in and make sure you have read access.')
      raise e
