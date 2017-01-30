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

"""Google Cloud Platform library - Generic SQL Helpers."""
from __future__ import absolute_import
from __future__ import unicode_literals
import types


# Names used for the arg parser, unnamed (main) query and last query in the module.
# Note that every module has a last query, but not every module has a main query.

_SQL_MODULE_ARGPARSE = '_sql_module_arg_parser'
_SQL_MODULE_MAIN = '_sql_module_main'
_SQL_MODULE_LAST = '_sql_module_last'


def get_default_query_from_module(module):
  """ Given a %%sql module return the default (last) query for the module.

  Args:
    module: the %%sql module.

  Returns:
    The default query associated with this module.
  """
  if isinstance(module, types.ModuleType):
    return module.__dict__.get(_SQL_MODULE_LAST, None)
  return None

