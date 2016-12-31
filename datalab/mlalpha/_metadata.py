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

import yaml

import google.cloud.ml as ml


class Metadata(object):
  """Helper class for parsing and serving feature metadata.
  """

  def __init__(self, metadata_path):
    """Initializes an instance of Metadata.

    Args:
      metadata_path: metadata file path. Can be local or GCS path.
    """
    self._metadata_path = metadata_path
    self._labels = {}

  def get_classification_label(self, label_index):
    """Get classification label given a label index.

    Args:
      label_index: the index of label.

    Returns:
      The classification label, or label_index if the metadata is not for classification.

    Raises:
      Exception if metadata is malformed.
    """
    if len(self._labels) == 0:
      with ml.util._file.open_local_or_gcs(self._metadata_path, 'r') as f:
        metadata = yaml.load(f)
        if 'columns' not in metadata:
          raise Exception('Invalid metadata. No columns found.')
        for column_name, column in metadata['columns'].iteritems():
          scenario = column.get('scenario', None)
          # classification is the old name and now is called discrete.
          if scenario == 'classification' or scenario == 'discrete':
            if 'items' not in column:
              raise Exception('Invalid metadata. No "items" found for "%s"' % column_name)
            for label, index in column['items'].iteritems():
              self._labels[index] = label
            break
          elif scenario is not None:
            return label_index # target column found but not classification
      if len(self._labels) == 0:
        raise Exception('Invalid metadata. No classification labels found.')
    return self._labels[label_index]

  def get_target_name_and_scenario(self):
    """Get name of the target feature and scenario.

    Returns:
      Name of the target feature or scenario

    Raises:
      Exception if metadata is malformed.
    """
    with ml.util._file.open_local_or_gcs(self._metadata_path, 'r') as f:
      metadata = yaml.load(f)
    if 'features' not in metadata or 'columns' not in metadata:
      raise Exception('Invalid metadata. No features or columns found.')
    target_column_name, scenario = None, None
    for column_name, column in metadata['columns'].iteritems():
      if 'scenario' in column:
        target_column_name, scenario = column_name, column['scenario']
        break
    if target_column_name is None:
      raise Exception('Invalid metadata. No target found in columns.')
    for feature_name, feature in metadata['features'].iteritems():
      if feature['columns'][0] == target_column_name:
        return feature_name, scenario
    raise Exception('Invalid metadata. No target found in features.')
