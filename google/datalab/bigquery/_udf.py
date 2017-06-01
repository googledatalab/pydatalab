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

"""Google Cloud Platform library - BigQuery UDF Functionality."""
from __future__ import absolute_import
from __future__ import unicode_literals
from past.builtins import basestring
from builtins import object


class UDF(object):
  """Represents a BigQuery UDF declaration.
  """

  @property
  def name(self):
    return self._name

  @property
  def imports(self):
    return self._imports

  @property
  def code(self):
    return self._code

  def __init__(self, name, code, return_type, params=None, language='js', imports=None):
    """Initializes a UDF object from its pieces.

    Args:
      name: the name of the javascript function
      code: function body implementing the logic.
      return_type: BigQuery data type of the function return. See supported data types in
        the BigQuery docs
      params: list of parameter tuples: (name, type)
      language: see list of supported languages in the BigQuery docs
      imports: a list of GCS paths containing further support code.
      """
    if not isinstance(return_type, basestring):
      raise TypeError('Argument return_type should be a string. Instead got: ', type(return_type))
    if params and not isinstance(params, list):
      raise TypeError('Argument params should be a list of parameter names and types')
    if imports and not isinstance(imports, list):
      raise TypeError('Argument imports should be a list of GCS string paths')
    if imports and language != 'js':
      raise Exception('Imports are available for Javascript UDFs only')

    self._name = name
    self._code = code
    self._return_type = return_type
    self._params = params or []
    self._language = language
    self._imports = imports or []
    self._sql = None

  def _expanded_sql(self):
    """Get the expanded BigQuery SQL string of this UDF

    Returns
      The expanded SQL string of this UDF
    """
    if not self._sql:
      self._sql = UDF._build_udf(self._name, self._code, self._return_type, self._params,
                                 self._language, self._imports)
    return self._sql

  def _repr_sql_(self):
    return self._expanded_sql()

  def __repr__(self):
    return 'BigQuery UDF - code:\n%s' % self._code

  @staticmethod
  def _build_udf(name, code, return_type, params, language, imports):
    """Creates the UDF part of a BigQuery query using its pieces

    Args:
      name: the name of the javascript function
      code: function body implementing the logic.
      return_type: BigQuery data type of the function return. See supported data types in
        the BigQuery docs
      params: dictionary of parameter names and types
      language: see list of supported languages in the BigQuery docs
      imports: a list of GCS paths containing further support code.
      """

    params = ','.join(['%s %s' % named_param for named_param in params])
    imports = ','.join(['library="%s"' % i for i in imports])

    udf = 'CREATE TEMPORARY FUNCTION {name} ({params})\n' +\
          'RETURNS {return_type}\n' +\
          'LANGUAGE {language}\n' +\
          'AS """\n' +\
          '{code}\n' +\
          '"""\n' +\
          'OPTIONS (\n' +\
          '{imports}\n' +\
          ');'
    return udf.format(name=name, params=params, return_type=return_type,
                      language=language, code=code, imports=imports)
