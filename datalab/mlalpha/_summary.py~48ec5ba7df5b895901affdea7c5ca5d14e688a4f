# Copyright 2016 Google Inc. All rights reserved.
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

"""Implements Cloud ML Summary wrapper."""

import datetime
import glob
import os
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record

import datalab.storage as storage


class Summary(object):
  """Represents TensorFlow summary events from files under a directory."""

  def __init__(self, path):
    """Initializes an instance of a Summary.

    Args:
      path: the path of the directory which holds TensorFlow events files.
            Can be local path or GCS path.
    """
    self._path = path

  def _get_events_files(self):
    if self._path.startswith('gs://'):
      storage._api.Api.verify_permitted_to_read(self._path)
      bucket, prefix = storage._bucket.parse_name(self._path)
      items = storage.Items(bucket, prefix, None)
      filtered_list = [item.uri for item in items if os.path.basename(item.uri).find('tfevents')]
      return filtered_list
    else:
      path_pattern = os.path.join(self._path, '*tfevents*')
      return glob.glob(path_pattern)

  def list_events(self):
    """List all scalar events in the directory.

    Returns:
      A set of unique event tags.
    """
    event_tags = set()
    for event_file in self._get_events_files():
      for record in tf_record.tf_record_iterator(event_file):
        event = event_pb2.Event.FromString(record)
        if event.summary is None or event.summary.value is None:
          continue
        for value in event.summary.value:
          if value.simple_value is None:
            continue
          if value.tag is not None and value.tag not in event_tags:
            event_tags.add(value.tag)
    return event_tags

  def get_events(self, event_name):
    """Get all events of a certain tag.

    Args:
      event_name: the tag of event to look for.

    Returns:
      A tuple. First is a list of {time_span, event_name}. Second is a list of {step, event_name}.

    Raises:
      Exception if event start time cannot be found
    """
    events_time = []
    events_step = []
    event_start_time = None
    for event_file in self._get_events_files():
      for record in tf_record.tf_record_iterator(event_file):
        event = event_pb2.Event.FromString(record)
        if event.file_version is not None:
          # first event in the file.
          time = datetime.datetime.fromtimestamp(event.wall_time)
          if event_start_time is None or event_start_time > time:
            event_start_time = time

        if event.summary is None or event.summary.value is None:
          continue
        for value in event.summary.value:
          if value.simple_value is None or value.tag is None:
            continue
          if value.tag == event_name:
            if event.wall_time is not None:
              time = datetime.datetime.fromtimestamp(event.wall_time)
              events_time.append({'time': time, event_name: value.simple_value})
            if event.step is not None:
              events_step.append({'step': event.step, event_name: value.simple_value})
    if event_start_time is None:
      raise Exception('Empty or invalid TF events file. Cannot find event start time.')
    for event in events_time:
      event['time'] = event['time'] - event_start_time  # convert time to timespan
    events_time = sorted(events_time, key=lambda k: k['time'])
    events_step = sorted(events_step, key=lambda k: k['step'])
    return events_time, events_step
