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

"""Useful common utility functions."""
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import str
from past.builtins import basestring

import collections
import datetime
import re


DatasetName = collections.namedtuple('DatasetName', ['project_id', 'dataset_id'])
""" A namedtuple for Dataset names.

  Args:
    project_id: the project id for the dataset.
    dataset_id: the dataset id for the dataset.
"""

TableName = collections.namedtuple('TableName',
                                   ['project_id', 'dataset_id', 'table_id', 'decorator'])
""" A namedtuple for Table names.

  Args:
    project_id: the project id for the table.
    dataset_id: the dataset id for the table.
    table_id: the table id for the table.
    decorator: the optional decorator for the table (for windowing/snapshot-ing).
"""

# Absolute project-qualified name pattern: <project>.<dataset>
_ABS_DATASET_NAME_PATTERN = r'^([a-z\d\-_\.:]+)\.(\w+)$'

# Relative name pattern: <dataset>
_REL_DATASET_NAME_PATTERN = r'^(\w+)$'

# Absolute project-qualified name pattern: <project>.<dataset>.<table>
_ABS_TABLE_NAME_PATTERN = r'^([a-z\d\-_\.:]+)\.(\w+)\.(\w+)(@[\d\-]+)?$'

# Relative name pattern: <dataset>.<table>
_REL_TABLE_NAME_PATTERN = r'^(\w+)\.(\w+)(@[\d\-]+)?$'

# Table-only name pattern: <table>. Includes an optional decorator.
_TABLE_NAME_PATTERN = r'^(\w+)(@[\d\-]+)$'


def parse_dataset_name(name, project_id=None):
  """Parses a dataset name into its individual parts.

  Args:
    name: the name to parse, or a tuple, dictionary or array containing the parts.
    project_id: the expected project ID. If the name does not contain a project ID,
        this will be used; if the name does contain a project ID and it does not match
        this, an exception will be thrown.
  Returns:
    A DatasetName named tuple for the dataset.
  Raises:
    Exception: raised if the name doesn't match the expected formats or a project_id was
        specified that does not match that in the name.
  """
  _project_id = _dataset_id = None
  if isinstance(name, basestring):
    # Try to parse as absolute name first.
    m = re.match(_ABS_DATASET_NAME_PATTERN, name, re.IGNORECASE)
    if m is not None:
      _project_id, _dataset_id = m.groups()
    else:
      # Next try to match as a relative name implicitly scoped within current project.
      m = re.match(_REL_DATASET_NAME_PATTERN, name)
      if m is not None:
        groups = m.groups()
        _dataset_id = groups[0]
  elif isinstance(name, dict):
    try:
      _dataset_id = name['dataset_id']
      _project_id = name['project_id']
    except KeyError:
      pass
  else:
    # Try treat as an array or tuple
    if len(name) == 2:
      # Treat as a tuple or array.
      _project_id, _dataset_id = name
    elif len(name) == 1:
      _dataset_id = name[0]
  if not _dataset_id:
    raise Exception('Invalid dataset name: ' + str(name))
  if not _project_id:
    _project_id = project_id

  return DatasetName(_project_id, _dataset_id)


def parse_table_name(name, project_id=None, dataset_id=None):
  """Parses a table name into its individual parts.

  Args:
    name: the name to parse, or a tuple, dictionary or array containing the parts.
    project_id: the expected project ID. If the name does not contain a project ID,
        this will be used; if the name does contain a project ID and it does not match
        this, an exception will be thrown.
    dataset_id: the expected dataset ID. If the name does not contain a dataset ID,
        this will be used; if the name does contain a dataset ID and it does not match
        this, an exception will be thrown.
  Returns:
    A TableName named tuple consisting of the full name and individual name parts.
  Raises:
    Exception: raised if the name doesn't match the expected formats, or a project_id and/or
        dataset_id was provided that does not match that in the name.
  """
  _project_id = _dataset_id = _table_id = _decorator = None
  if isinstance(name, basestring):
    # Try to parse as absolute name first.
    m = re.match(_ABS_TABLE_NAME_PATTERN, name, re.IGNORECASE)
    if m is not None:
      _project_id, _dataset_id, _table_id, _decorator = m.groups()
    else:
      # Next try to match as a relative name implicitly scoped within current project.
      m = re.match(_REL_TABLE_NAME_PATTERN, name)
      if m is not None:
        groups = m.groups()
        _project_id, _dataset_id, _table_id, _decorator =\
            project_id, groups[0], groups[1], groups[2]
      else:
        # Finally try to match as a table name only.
        m = re.match(_TABLE_NAME_PATTERN, name)
        if m is not None:
          groups = m.groups()
          _project_id, _dataset_id, _table_id, _decorator =\
              project_id, dataset_id, groups[0], groups[1]
  elif isinstance(name, dict):
    try:
      _table_id = name['table_id']
      _dataset_id = name['dataset_id']
      _project_id = name['project_id']
    except KeyError:
      pass
  else:
    # Try treat as an array or tuple
    if len(name) == 4:
      _project_id, _dataset_id, _table_id, _decorator = name
    elif len(name) == 3:
      _project_id, _dataset_id, _table_id = name
    elif len(name) == 2:
      _dataset_id, _table_id = name
  if not _table_id:
    raise Exception('Invalid table name: ' + str(name))
  if not _project_id:
    _project_id = project_id
  if not _dataset_id:
    _dataset_id = dataset_id
  if not _decorator:
    _decorator = ''

  return TableName(_project_id, _dataset_id, _table_id, _decorator)


def format_query_errors(errors):
  return '\n'.join(['%s: %s' % (error['reason'], error['message']) for error in errors])


def get_query_parameters_internal(config_parameters):
  # We merge the parameters with the airflow macros so that users can specify certain airflow
  # macro names (like '@ds') in their sql. This is useful for enabling the user to progressively
  # author a pipeline with %bq pipeline.
  today = datetime.date.today()
  now = datetime.datetime.now()
  default_query_parameters = {
    # the datetime formatted as YYYY-MM-DD
    '_ds': {'type': 'STRING', 'value': today.isoformat()},
    # the full ISO-formatted timestamp YYYY-MM-DDTHH:MM:SS.mmmmmm
    '_ts': {'type': 'STRING', 'value': now.isoformat()},
    # the datetime formatted as YYYYMMDD (i.e. YYYY-MM-DD with 'no dashes')
    '_ds_nodash': {'type': 'STRING', 'value': today.strftime('%Y%m%d')},
    # the timestamp formatted as YYYYMMDDTHHMMSSmmmmmm (i.e full ISO-formatted timestamp
    # YYYY-MM-DDTHH:MM:SS.mmmmmm with no dashes or colons).
    '_ts_nodash': {'type': 'STRING', 'value': now.strftime('%Y%m%d%H%M%S%f')},
    '_ts_year': {'type': 'STRING', 'value': today.strftime('%Y')},
    '_ts_month': {'type': 'STRING', 'value': today.strftime('%m')},
    '_ts_day': {'type': 'STRING', 'value': today.strftime('%d')},
    '_ts_hour': {'type': 'STRING', 'value': now.strftime('%H')},
    '_ts_minute': {'type': 'STRING', 'value': now.strftime('%M')},
    '_ts_second': {'type': 'STRING', 'value': now.strftime('%S')},
  }
  # We merge the parameters by first pushing in the query parameters into a dictionary keyed by
  # the parameter name. We then use this to update the canned parameters dictionary. This will
  # have the effect of using user-provided parameters in case of naming conflicts.
  config_parameters = config_parameters or {}
  input_query_parameters = {
    item['name']: {
      'type': item['type'],
      'value': item['value']
    } for item in config_parameters
  }
  merged_parameters = default_query_parameters.copy()
  merged_parameters.update(input_query_parameters)
  # Parse query_params. We're exposing a simpler schema format than the one actually required
  # by BigQuery to make magics easier. We need to convert between the two formats
  parsed_params = []
  for key, value in merged_parameters.items():
    parsed_params.append({
      'name': key,
      'parameterType': {
        'type': value['type']
      },
      'parameterValue': {
        'value': value['value']
      }
    })
  return parsed_params
