# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Classes for dealing with I/O from ML pipelines.
"""

import json

import yaml


class JsonCoder(object):
  """A coder to encode and decode JSON formatted data."""

  def __init__(self, indent=None):
    self._indent = indent

  def encode(self, obj):
    """Encodes a python object into a JSON string.

    Args:
      obj: python object.

    Returns:
      JSON string.
    """
    # Supplying seperators to avoid unnecessary trailing whitespaces.
    return json.dumps(obj, indent=self._indent, separators=(',', ': '))

  def decode(self, json_string):
    """Decodes a JSON string to a python object.

    Args:
      json_string: A JSON string.

    Returns:
      A python object.
    """
    return json.loads(json_string)


class YamlCoder(object):
  """A coder to encode and decode YAML formatted data."""

  def __init__(self):
    """Trying to use the efficient libyaml library to encode and decode yaml.

    If libyaml is not available than we fallback to use the native yaml library,
    use with caution; it is far less efficient, uses excessive memory, and leaks
    memory.
    """
    if yaml.__with_libyaml__:
      self._safe_dumper = yaml.CSafeDumper
      self._safe_loader = yaml.CSafeLoader
    else:
      self._safe_dumper = yaml.SafeDumper
      self._safe_loader = yaml.SafeLoader

  def encode(self, obj):
    """Encodes a python object into a YAML string.

    Args:
      obj: python object.

    Returns:
      YAML string.
    """
    return yaml.dump(
        obj,
        default_flow_style=False,
        encoding='utf-8',
        Dumper=self._safe_dumper)

  def decode(self, yaml_string):
    """Decodes a YAML string to a python object.

    Args:
      yaml_string: A YAML string.

    Returns:
      A python object.
    """
    return yaml.load(yaml_string, Loader=self._safe_loader)
