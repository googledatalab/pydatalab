# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections
import datetime
import fnmatch
import matplotlib.pyplot as plt
import os
import pandas as pd
import six
import tensorflow as tf

from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record


class Summary(object):
  """Represents TensorFlow summary events from files under specified directories."""

  def __init__(self, paths):
    """Initializes an instance of a Summary.
    Args:
      path: a path or a list of paths to directories which hold TensorFlow events files.
            Can be local path or GCS paths. Wild cards allowed.
    """

    if isinstance(paths, six.string_types):
      self._paths = [paths]
    else:
      self._paths = paths

  def _glob_events_files(self, paths, recursive):
    """Find all tf events files under a list of paths recursively. """

    event_files = []
    for path in paths:
      dirs = tf.gfile.Glob(path)
      dirs = filter(lambda x: tf.gfile.IsDirectory(x), dirs)
      for dir in dirs:
        if recursive:
          dir_files_pair = [(root, filenames) for root, _, filenames in tf.gfile.Walk(dir)]
        else:
          dir_files_pair = [(dir, tf.gfile.ListDirectory(dir))]

        for root, filenames in dir_files_pair:
          file_names = fnmatch.filter(filenames, '*.tfevents.*')
          file_paths = [os.path.join(root, x) for x in file_names]
          file_paths = filter(lambda x: not tf.gfile.IsDirectory(x), file_paths)
          event_files += file_paths
    return event_files

  def list_events(self):
    """List all scalar events in the directory.

    Returns:
      A dictionary. Key is the name of a event. Value is a set of dirs that contain that event.
    """
    event_dir_dict = collections.defaultdict(set)

    for event_file in self._glob_events_files(self._paths, recursive=True):
      dir = os.path.dirname(event_file)
      try:
        for record in tf_record.tf_record_iterator(event_file):
          event = event_pb2.Event.FromString(record)
          if event.summary is None or event.summary.value is None:
            continue
          for value in event.summary.value:
            if value.simple_value is None or value.tag is None:
              continue
            event_dir_dict[value.tag].add(dir)
      except tf.errors.DataLossError:
        # DataLossError seems to happen sometimes for small logs.
        # We want to show good records regardless.
        continue
    return dict(event_dir_dict)

  def get_events(self, event_names):
    """Get all events as pandas DataFrames given a list of names.
    Args:
      event_names: A list of events to get.
    Returns:
      A list with the same length and order as event_names. Each element is a dictionary
          {dir1: DataFrame1, dir2: DataFrame2, ...}.
          Multiple directories may contain events with the same name, but they are different
          events (i.e. 'loss' under trains_set/, and 'loss' under eval_set/.)
    """

    if isinstance(event_names, six.string_types):
      event_names = [event_names]

    all_events = self.list_events()
    dirs_to_look = set()
    for event, dirs in six.iteritems(all_events):
      if event in event_names:
        dirs_to_look.update(dirs)

    ret_events = [collections.defaultdict(lambda: pd.DataFrame(columns=['time', 'step', 'value']))
                  for i in range(len(event_names))]
    for event_file in self._glob_events_files(dirs_to_look, recursive=False):
      try:
        for record in tf_record.tf_record_iterator(event_file):
          event = event_pb2.Event.FromString(record)
          if event.summary is None or event.wall_time is None or event.summary.value is None:
            continue

          event_time = datetime.datetime.fromtimestamp(event.wall_time)
          for value in event.summary.value:
            if value.tag not in event_names or value.simple_value is None:
              continue

            index = event_names.index(value.tag)
            dir_event_dict = ret_events[index]
            dir = os.path.dirname(event_file)
            # Append a row.
            df = dir_event_dict[dir]
            df.loc[len(df)] = [event_time, event.step, value.simple_value]
      except tf.errors.DataLossError:
        # DataLossError seems to happen sometimes for small logs.
        # We want to show good records regardless.
        continue

    for idx, dir_event_dict in enumerate(ret_events):
      for df in dir_event_dict.values():
        df.sort_values(by=['time'], inplace=True)
      ret_events[idx] = dict(dir_event_dict)

    return ret_events

  def plot(self, event_names, x_axis='step'):
    """Plots a list of events. Each event (a dir+event_name) is represetented as a line
       in the graph.
    Args:
      event_names: A list of events to plot. Each event_name may correspond to multiple events,
          each in a different directory.
      x_axis: whether to use step or time as x axis.
    """

    if isinstance(event_names, six.string_types):
      event_names = [event_names]

    events_list = self.get_events(event_names)
    for event_name, dir_event_dict in zip(event_names, events_list):
      for dir, df in six.iteritems(dir_event_dict):
        label = event_name + ':' + dir
        x_column = df['step'] if x_axis == 'step' else df['time']
        plt.plot(x_column, df['value'], label=label)
    plt.legend(loc='best')
    plt.show()
