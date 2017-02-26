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

"""Google Cloud Platform library - BigQuery IPython Functionality."""
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import str
from past.builtins import basestring

try:
  import IPython
  import IPython.core.display
  import IPython.core.magic
except ImportError:
  raise Exception('This module can only be loaded in ipython.')

import fnmatch
import json
import re

import google.datalab.storage
import google.datalab.utils.commands


def _extract_gcs_api_response_error(message):
  """ A helper function to extract user-friendly error messages from service exceptions.

  Args:
    message: An error message from an exception. If this is from our HTTP client code, it
        will actually be a tuple.

  Returns:
    A modified version of the message that is less cryptic.
  """
  try:
    if len(message) == 3:
      # Try treat the last part as JSON
      data = json.loads(message[2])
      return data['error']['errors'][0]['message']
  except Exception:
    pass
  return message


@IPython.core.magic.register_line_cell_magic
def gcs(line, cell=None):
  """Implements the gcs cell magic for ipython notebooks.

  Args:
    line: the contents of the gcs line.
  Returns:
    The results of executing the cell.
  """
  parser = google.datalab.utils.commands.CommandParser(prog='gcs', description="""
Execute various Google Cloud Storage related operations. Use "%gcs <command> -h"
for help on a specific command.
""")

  # TODO(gram): consider adding a move command too. I did try this already using the
  # objects.patch API to change the object name but that fails with an error:
  #
  # Value 'newname' in content does not agree with value 'oldname'. This can happen when a value
  # set through a parameter is inconsistent with a value set in the request.
  #
  # This is despite 'name' being identified as writable in the storage API docs.
  # The alternative would be to use a copy/delete.
  copy_parser = parser.subcommand('copy',
                                  'Copy one or more Google Cloud Storage objects to a different location.')
  copy_parser.add_argument('-s', '--source', help='The name of the object(s) to copy', nargs='+')
  copy_parser.add_argument('-d', '--destination', required=True,
      help='The copy destination. For multiple source objects this must be a bucket.')
  copy_parser.set_defaults(func=_gcs_copy)

  create_parser = parser.subcommand('create', 'Create one or more Google Cloud Storage buckets.')
  create_parser.add_argument('-p', '--project', help='The project associated with the objects')
  create_parser.add_argument('-b', '--bucket', help='The name of the bucket(s) to create',
                             nargs='+')
  create_parser.set_defaults(func=_gcs_create)

  delete_parser = parser.subcommand('delete', 'Delete one or more Google Cloud Storage buckets or objects.')
  delete_parser.add_argument('-b', '--bucket', nargs='*',
                             help='The name of the bucket(s) to remove')
  delete_parser.add_argument('-o', '--object', nargs='*',
                             help='The name of the object(s) to remove')
  delete_parser.set_defaults(func=_gcs_delete)

  list_parser = parser.subcommand('list', 'List buckets in a project, or contents of a bucket.')
  list_parser.add_argument('-p', '--project', help='The project associated with the objects')
  list_parser.add_argument('-o', '--objects',
                     help='List objects under the given Google Cloud Storage path',
                     nargs='?')
  list_parser.set_defaults(func=_gcs_list)

  read_parser = parser.subcommand('read',
                                  'Read the contents of a Google Cloud Storage object into a Python variable.')
  read_parser.add_argument('-o', '--object', help='The name of the object to read',
                           required=True)
  read_parser.add_argument('-v', '--variable', required=True,
                           help='The name of the Python variable to set')
  read_parser.set_defaults(func=_gcs_read)

  view_parser = parser.subcommand('view', 'View the contents of a Google Cloud Storage object.')
  view_parser.add_argument('-n', '--head', type=int, default=20,
                           help='The number of initial lines to view')
  view_parser.add_argument('-t', '--tail', type=int, default=20,
                           help='The number of lines from end to view')
  view_parser.add_argument('-o', '--object', help='The name of the object to view',
                           required=True)
  view_parser.set_defaults(func=_gcs_view)

  write_parser = parser.subcommand('write',
                                   'Write the value of a Python variable to a Google Cloud Storage object.')
  write_parser.add_argument('-v', '--variable', help='The name of the source Python variable',
                            required=True)
  write_parser.add_argument('-o', '--object', required=True,
                            help='The name of the destination Google Cloud Storage object to write')
  write_parser.add_argument('-c', '--content_type', help='MIME type', default='text/plain')
  write_parser.set_defaults(func=_gcs_write)

  return google.datalab.utils.commands.handle_magic_line(line, cell, parser)


def _parser_exit(status=0, message=None):
  """ Replacement exit method for argument parser. We want to stop processing args but not
      call sys.exit(), so we raise an exception here and catch it in the call to parse_args.
  """
  raise Exception()


def _expand_list(names):
  """ Do a wildchar name expansion of object names in a list and return expanded list.

    The objects are expected to exist as this is used for copy sources or delete targets.
    Currently we support wildchars in the key name only.
  """

  if names is None:
    names = []
  elif isinstance(names, basestring):
    names = [names]

  results = []  # The expanded list.
  objects = {}  # Cached contents of buckets; used for matching.
  for name in names:
    bucket, key = google.datalab.storage._bucket.parse_name(name)
    results_len = len(results)  # If we fail to add any we add name and let caller deal with it.
    if bucket:
      if not key:
        # Just a bucket; add it.
        results.append('gs://%s' % bucket)
      elif google.datalab.storage.Object(bucket, key).exists():
        results.append('gs://%s/%s' % (bucket, key))
      else:
        # Expand possible key values.
        if bucket not in objects and key[:1] == '*':
          # We need the full list; cache a copy for efficiency.
          objects[bucket] = [obj.metadata.name for obj in list(google.datalab.storage.Bucket(bucket).objects())]
        # If we have a cached copy use it
        if bucket in objects:
          candidates = objects[bucket]
        # else we have no cached copy but can use prefix matching which is more efficient than
        # getting the full contents.
        else:
          # Get the non-wildchar prefix.
          match = re.search('\?|\*|\[', key)
          prefix = key
          if match:
            prefix = key[0:match.start()]

          candidates = [obj.metadata.name
                        for obj in google.datalab.storage.Bucket(bucket).objects(prefix=prefix)]

        for obj in candidates:
          if fnmatch.fnmatch(obj, key):
            results.append('gs://%s/%s' % (bucket, obj))

    # If we added no matches, add the original name and let caller deal with it.
    if len(results) == results_len:
      results.append(name)

  return results


def _gcs_copy(args, _):
  target = args['destination']
  target_bucket, target_key = google.datalab.storage._bucket.parse_name(target)
  if target_bucket is None and target_key is None:
    raise Exception('Invalid copy target name %s' % target)

  sources = _expand_list(args['source'])

  if len(sources) > 1:
    # Multiple sources; target must be a bucket
    if target_bucket is None or target_key is not None:
      raise Exception('More than one source but target %s is not a bucket' % target)

  errs = []
  for source in sources:
    source_bucket, source_key = google.datalab.storage._bucket.parse_name(source)
    if source_bucket is None or source_key is None:
      raise Exception('Invalid source object name %s' % source)
    destination_bucket = target_bucket if target_bucket else source_bucket
    destination_key = target_key if target_key else source_key
    try:
      google.datalab.storage.Object(source_bucket, source_key).copy_to(destination_key,
                                                          bucket=destination_bucket)
    except Exception as e:
      errs.append("Couldn't copy %s to %s: %s" %
                  (source, target, _extract_gcs_api_response_error(str(e))))
  if errs:
    raise Exception('\n'.join(errs))


def _gcs_create(args, _):
  """ Create one or more buckets. """
  errs = []
  for name in args['bucket']:
    try:
      bucket, key = google.datalab.storage._bucket.parse_name(name)
      if bucket and not key:
        google.datalab.storage.Bucket(bucket).create(_make_context(args['project']))
      else:
        raise Exception("Invalid bucket name %s" % name)
    except Exception as e:
      errs.append("Couldn't create %s: %s" %
                  (name, _extract_gcs_api_response_error(str(e))))
  if errs:
    raise Exception('\n'.join(errs))


def _gcs_delete(args, _):
  """ Delete one or more buckets or objects. """
  objects = _expand_list(args['bucket'])
  objects.extend(_expand_list(args['object']))
  errs = []
  for obj in objects:
    try:
      bucket, key = google.datalab.storage._bucket.parse_name(obj)
      if bucket and key:
        gcs_object = google.datalab.storage.Object(bucket, key)
        if gcs_object.exists():
          google.datalab.storage.Object(bucket, key).delete()
        else:
          errs.append("%s does not exist" % obj)
      elif bucket:
        gcs_bucket = google.datalab.storage.Bucket(bucket)
        if gcs_bucket.exists():
          gcs_bucket.delete()
        else:
          errs.append("%s does not exist" % obj)
      else:
        raise Exception("Can't delete object with invalid name %s" % obj)
    except Exception as e:
      errs.append("Couldn't delete %s: %s" %
                  (obj, _extract_gcs_api_response_error(str(e))))
  if errs:
    raise Exception('\n'.join(errs))

def _make_context(project_id=None):
  default_context = google.datalab.Context.default()
  project_id = project_id or default_context.project_id
  return google.datalab.Context(project_id, default_context.credentials)

def _gcs_list_buckets(project, pattern):
  """ List all Google Cloud Storage buckets that match a pattern. """
  data = [{'Bucket': 'gs://' + bucket.name, 'Created': bucket.metadata.created_on}
          for bucket in google.datalab.storage.Buckets(_make_context(project))
          if fnmatch.fnmatch(bucket.name, pattern)]
  return google.datalab.utils.commands.render_dictionary(data, ['Bucket', 'Created'])


def _gcs_get_keys(bucket, pattern):
  """ Get names of all Google Cloud Storage keys in a specified bucket that match a pattern. """
  return [obj for obj in list(bucket.objects()) if fnmatch.fnmatch(obj.metadata.name, pattern)]


def _gcs_get_key_names(bucket, pattern):
  """ Get names of all Google Cloud Storage keys in a specified bucket that match a pattern. """
  return [obj.metadata.name for obj in _gcs_get_keys(bucket, pattern)]


def _gcs_list_keys(bucket, pattern):
  """ List all Google Cloud Storage keys in a specified bucket that match a pattern. """
  data = [{'Name': obj.metadata.name,
           'Type': obj.metadata.content_type,
           'Size': obj.metadata.size,
           'Updated': obj.metadata.updated_on}
          for obj in _gcs_get_keys(bucket, pattern)]
  return google.datalab.utils.commands.render_dictionary(data, ['Name', 'Type', 'Size', 'Updated'])


def _gcs_list(args, _):
  """ List the buckets or the contents of a bucket.

  This command is a bit different in that we allow wildchars in the bucket name and will list
  the buckets that match.
  """
  target = args['objects']
  project = args['project']
  if target is None:
    return _gcs_list_buckets(project, '*')  # List all buckets.

  bucket_name, key = google.datalab.storage._bucket.parse_name(target)
  if bucket_name is None:
    raise Exception('Cannot list %s; not a valid bucket name' % target)

  # If a target was specified, list keys inside it
  if target:
    if not re.search('\?|\*|\[', target):
      # If no wild characters are present in the key string, append a '/*' suffix to show all keys
      key = key.strip('/') + '/*' if key else '*'

    if project:
      # Only list if the bucket is in the project
      for bucket in google.datalab.storage.Buckets(_make_context(project)):
        if bucket.name == bucket_name:
          break
      else:
        raise Exception('%s does not exist in project %s' % (target, project))
    else:
      bucket = google.datalab.storage.Bucket(bucket_name)

    if bucket.exists():
      return _gcs_list_keys(bucket, key)
    else:
      raise Exception('Bucket %s does not exist' % target)

  else:
    # Treat the bucket name as a pattern and show matches. We don't use bucket_name as that
    # can strip off wildchars and so we need to strip off gs:// here.
    return _gcs_list_buckets(project, target.strip('/')[5:])


def _get_object_contents(source_name):
  source_bucket, source_key = google.datalab.storage._bucket.parse_name(source_name)
  if source_bucket is None:
    raise Exception('Invalid source object name %s; no bucket specified.' % source_name)
  if source_key is None:
    raise Exception('Invalid source object name %si; source cannot be a bucket.' % source_name)
  source = google.datalab.storage.Object(source_bucket, source_key)
  if not source.exists():
    raise Exception('Source object %s does not exist' % source_name)
  return source.download()


def _gcs_read(args, _):
  contents = _get_object_contents(args['object'])
  ipy = IPython.get_ipython()
  ipy.push({args['variable']: contents})


def _gcs_view(args, _):
  contents = _get_object_contents(args['object'])
  if not isinstance(contents, basestring):
    contents = str(contents)
  lines = contents.split('\n')
  head_count = args['head']
  tail_count = args['tail']
  if len(lines) > head_count + tail_count:
    head = '\n'.join(lines[:head_count])
    tail = '\n'.join(lines[-tail_count:])
    return head + '\n...\n' + tail
  else:
    return contents


def _gcs_write(args, _):
  target_name = args['object']
  target_bucket, target_key = google.datalab.storage._bucket.parse_name(target_name)
  if target_bucket is None or target_key is None:
    raise Exception('Invalid target object name %s' % target_name)
  target = google.datalab.storage.Object(target_bucket, target_key)
  ipy = IPython.get_ipython()
  contents = ipy.user_ns[args['variable']]
  # TODO(gram): would we want to to do any special handling here; e.g. for DataFrames?
  target.write_stream(str(contents), args['content_type'])
