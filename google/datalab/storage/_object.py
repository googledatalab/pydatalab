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

"""Implements Object-related Cloud Storage APIs."""
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import object

import dateutil.parser
import logging
import time

import google.datalab
import google.datalab.utils

from . import _api

# TODO(nikhilko): Read/write operations don't account for larger files, or non-textual content.
#                 Use streaming reads into a buffer or StringIO or into a file handle.

# In some polling operations, we sleep between API calls to avoid hammering the
# server. This argument controls how long we sleep between API calls.
_POLLING_SLEEP = 1
# This argument controls how many times we'll poll before giving up.
_MAX_POLL_ATTEMPTS = 30


class ObjectMetadata(object):
  """Represents metadata about a Cloud Storage object."""

  def __init__(self, info):
    """Initializes an instance of a ObjectMetadata object.

    Args:
      info: a dictionary containing information about an Object.
    """
    self._info = info

  @property
  def content_type(self):
    """The Content-Type associated with the object, if any."""
    return self._info.get('contentType', None)

  @property
  def etag(self):
    """The ETag of the object, if any."""
    return self._info.get('etag', None)

  @property
  def name(self):
    """The name of the object."""
    return self._info['name']

  @property
  def size(self):
    """The size (in bytes) of the object. 0 for objects that don't exist."""
    return int(self._info.get('size', 0))

  @property
  def updated_on(self):
    """The updated timestamp of the object as a datetime.datetime."""
    s = self._info.get('updated', None)
    return dateutil.parser.parse(s) if s else None


class Object(object):
  """Represents a Cloud Storage object within a bucket."""

  def __init__(self, bucket, key, info=None, context=None):
    """Initializes an instance of an Object.

    Args:
      bucket: the name of the bucket containing the object.
      key: the key of the object.
      info: the information about the object if available.
      context: an optional Context object providing project_id and credentials. If a specific
          project id or credentials are unspecified, the default ones configured at the global
          level are used.
    """
    if context is None:
      context = google.datalab.Context.default()
    self._context = context
    self._api = _api.Api(context)
    self._bucket = bucket
    self._key = key
    self._info = info

  @staticmethod
  def from_url(url):
    from . import _bucket
    bucket, object = _bucket.parse_name(url)
    return Object(bucket, object)

  @property
  def key(self):
    """Returns the key of the object."""
    return self._key

  @property
  def uri(self):
    """Returns the gs:// URI for the object.
    """
    return 'gs://%s/%s' % (self._bucket, self._key)

  def __repr__(self):
    """Returns a representation for the table for showing in the notebook.
    """
    return 'Google Cloud Storage Object %s' % self.uri

  def copy_to(self, new_key, bucket=None):
    """Copies this object to the specified new key.

    Args:
      new_key: the new key to copy this object to.
      bucket: the bucket of the new object; if None (the default) use the same bucket.
    Returns:
      An Object corresponding to new key.
    Raises:
      Exception if there was an error copying the object.
    """
    if bucket is None:
      bucket = self._bucket
    try:
      new_info = self._api.objects_copy(self._bucket, self._key, bucket, new_key)
    except Exception as e:
      raise e
    return Object(bucket, new_key, new_info, context=self._context)

  def exists(self):
    """ Checks if the object exists. """
    try:
      return self.metadata is not None
    except google.datalab.utils.RequestException:
      return False
    except Exception as e:
      raise e

  def delete(self, wait_for_deletion=True):
    """Deletes this object from its bucket.

    Args:
      wait_for_deletion: If True, we poll until this object no longer appears in
          objects.list operations for this bucket before returning.

    Raises:
      Exception if there was an error deleting the object.
    """
    if self.exists():
      try:
        self._api.objects_delete(self._bucket, self._key)
      except Exception as e:
        raise e
      if wait_for_deletion:
        for _ in range(_MAX_POLL_ATTEMPTS):
          objects = Objects(self._bucket, prefix=self.key, delimiter='/',
                            context=self._context)
          if any(o.key == self.key for o in objects):
            time.sleep(_POLLING_SLEEP)
            continue
          break
        else:
          logging.error('Failed to see object deletion after %d attempts.',
                        _MAX_POLL_ATTEMPTS)

  @property
  def metadata(self):
    """Retrieves metadata about the object.

    Returns:
      An ObjectMetadata instance with information about this object.
    Raises:
      Exception if there was an error requesting the object's metadata.
    """
    if self._info is None:
      try:
        self._info = self._api.objects_get(self._bucket, self._key)
      except Exception as e:
        raise e
    return ObjectMetadata(self._info) if self._info else None

  def read_stream(self, start_offset=0, byte_count=None):
    """Reads the content of this object as text.

    Args:
      start_offset: the start offset of bytes to read.
      byte_count: the number of bytes to read. If None, it reads to the end.
    Returns:
      The text content within the object.
    Raises:
      Exception if there was an error requesting the object's content.
    """
    try:
      return self._api.object_download(self._bucket, self._key,
                                       start_offset=start_offset, byte_count=byte_count)
    except Exception as e:
      raise e

  def download(self):
    """Reads the content of this object.

    Returns:
      The content within the object.
    Raises:
      Exception if there was an error requesting the object's content.
    """
    return self.read_stream()

  def read_lines(self, max_lines=None):
    """Reads the content of this object as text, and return a list of lines up to some max.

    Args:
      max_lines: max number of lines to return. If None, return all lines.
    Returns:
      The text content of the object as a list of lines.
    Raises:
      Exception if there was an error requesting the object's content.
    """
    if max_lines is None:
      return self.read_stream().split('\n')

    max_to_read = self.metadata.size
    bytes_to_read = min(100 * max_lines, self.metadata.size)
    while True:
      content = self.read_stream(byte_count=bytes_to_read)

      lines = content.split('\n')
      if len(lines) > max_lines or bytes_to_read >= max_to_read:
        break
      # try 10 times more bytes or max
      bytes_to_read = min(bytes_to_read * 10, max_to_read)

    # remove the partial line at last
    del lines[-1]
    return lines[0:max_lines]

  def write_stream(self, content, content_type):
    """Writes text content to this object.

    Args:
      content: the text content to be written.
      content_type: the type of text content.
    Raises:
      Exception if there was an error requesting the object's content.
    """
    try:
      self._api.object_upload(self._bucket, self._key, content, content_type)
    except Exception as e:
      raise e

  def upload(self, content):
    """Uploads content to this object.

    Args:
      content: the text content to be written.
    Raises:
      Exception if there was an error requesting the object's content.
    """
    self.write_stream(content, content_type=None)


class Objects(object):
  """Represents a list of Cloud Storage objects within a bucket."""

  def __init__(self, bucket, prefix, delimiter, context=None):
    """Initializes an instance of an ObjectList.

    Args:
      bucket: the name of the bucket containing the objects.
      prefix: an optional prefix to match objects.
      delimiter: an optional string to simulate directory-like semantics. The returned objects
           will be those whose names do not contain the delimiter after the prefix. For
           the remaining objects, the names will be returned truncated after the delimiter
           with duplicates removed (i.e. as pseudo-directories).
      context: an optional Context object providing project_id and credentials. If a specific
          project id or credentials are unspecified, the default ones configured at the global
          level are used.
    """
    if context is None:
      context = google.datalab.Context.default()
    self._context = context
    self._api = _api.Api(context)
    self._bucket = bucket
    self._prefix = prefix
    self._delimiter = delimiter

  def contains(self, key):
    """Checks if the specified object exists.

    Args:
      key: the key of the object to lookup.
    Returns:
      True if the object exists; False otherwise.
    Raises:
      Exception if there was an error requesting information about the object.
    """
    try:
      self._api.objects_get(self._bucket, key)
    except google.datalab.utils.RequestException as e:
      if e.status == 404:
        return False
      raise e
    except Exception as e:
      raise e
    return True

  def _retrieve_objects(self, page_token, _):
    try:
      list_info = self._api.objects_list(self._bucket,
                                         prefix=self._prefix, delimiter=self._delimiter,
                                         page_token=page_token)
    except Exception as e:
      raise e

    objects = list_info.get('items', [])
    if len(objects):
      try:
        objects = [Object(self._bucket, info['name'], info, context=self._context)
                   for info in objects]
      except KeyError:
        raise Exception('Unexpected response from server')

    page_token = list_info.get('nextPageToken', None)
    return objects, page_token

  def __iter__(self):
    return iter(google.datalab.utils.Iterator(self._retrieve_objects))
