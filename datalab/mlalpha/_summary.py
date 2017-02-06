import datetime
import fnmatch
import glob
import google.cloud.ml as ml
import matplotlib.pyplot as plt
import os
import pandas as pd
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record


class Summary(object):
  """Represents TensorFlow summary events from files under specified directories."""

  def __init__(self, paths):
    """Initializes an instance of a Summary.

    Args:
      path: a list of paths to directories which hold TensorFlow events files.
            Can be local path or GCS paths. Wild cards allowed.
    """
    self._paths = [paths] if isinstance(paths, basestring) else paths

  def _glob_events_files(self, paths):
    event_files = []
    for path in paths:
      if path.startswith('gs://'):
        event_files += ml.util._file.glob_files(os.path.join(path, '*.tfevents.*'))
      else:
        dirs = ml.util._file.glob_files(path)
        for dir in dirs:
          for root, _, filenames in os.walk(dir):
            for filename in fnmatch.filter(filenames, '*.tfevents.*'):
              event_files.append(os.path.join(root, filename))
    return event_files

  def list_events(self):
    """List all scalar events in the directory.

    Returns:
      A dictionary. Key is the name of a event. Value is a set of dirs that contain that event.
    """
    event_dir_dict = {}
    for event_file in self._glob_events_files(self._paths):
      dir = os.path.dirname(event_file)
      for record in tf_record.tf_record_iterator(event_file):
        event = event_pb2.Event.FromString(record)
        if event.summary is None or event.summary.value is None:
          continue
        for value in event.summary.value:
          if value.simple_value is None or value.tag is None:
            continue
          if not value.tag in event_dir_dict:
            event_dir_dict[value.tag] = set()
          event_dir_dict[value.tag].add(dir)
    return event_dir_dict
  

  def get_events(self, event_names):
    """Get all events as pandas DataFrames given a list of names.

    Args:
      event_names: A list of events to get.

    Returns:
      A list with the same length as event_names. Each element is a dictionary
          {dir1: DataFrame1, dir2: DataFrame2, ...}.
          Multiple directories may contain events with the same name, but they are different
          events (i.e. 'loss' under trains_set/, and 'loss' under eval_set/.)
    """
    event_names = [event_names] if isinstance(event_names, basestring) else event_names

    all_events = self.list_events()
    dirs_to_look = set()
    for event, dirs in all_events.iteritems():
      if event in event_names:
        dirs_to_look.update(dirs)

    ret_events = [dict() for i in range(len(event_names))]
    for dir in dirs_to_look:
      for event_file in self._glob_events_files([dir]):
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
            if dir not in dir_event_dict:
              dir_event_dict[dir] = pd.DataFrame(
                  [[event_time, event.step, value.simple_value]],
                  columns=['time', 'step', 'value'])
            else:
              df = dir_event_dict[dir]
              # Append a row.
              df.loc[len(df)] = [event_time, event.step, value.simple_value]

    for dir_event_dict in ret_events:
      for df in dir_event_dict.values():
        df.sort_values(by=['time'], inplace=True)

    return ret_events

  def plot(self, event_names, x_axis='step'):
    """Plots a list of events. Each event (a dir+event_name) is represetented as a line
       in the graph.

    Args:
      event_names: A list of events to plot. Each event_name may correspond to multiple events,
          each in a different directory.
      x_axis: whether to use step or time as x axis.
    """
    event_names = [event_names] if isinstance(event_names, basestring) else event_names
    events_list = self.get_events(event_names)
    for event_name, dir_event_dict in zip(event_names, events_list):
      for dir, df in dir_event_dict.iteritems():
        label = event_name + ':' + dir
        x_column = df['step'] if x_axis == 'step' else df['time']
        plt.plot(x_column, df['value'], label=label)
    plt.legend(loc='best')
    plt.show()

