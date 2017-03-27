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


def _next_token(sql):
  """ This is a basic tokenizer for our limited purposes.
  It splits a SQL statement up into a series of segments, where a segment is one of:
  - identifiers
  - left or right parentheses
  - multi-line comments
  - single line comments
  - white-space sequences
  - string literals
  - consecutive strings of characters that are not one of the items above

  The aim is for us to be able to find function calls (identifiers followed by '('), and the
  associated closing ')') so we can augment these if needed.

  Args:
    sql: a SQL statement as a (possibly multi-line) string.

  Returns:
    For each call, the next token in the initial input.
  """
  i = 0

  # We use def statements here to make the logic more clear. The start_* functions return
  # true if i is the index of the start of that construct, while the end_* functions
  # return true if i point to the first character beyond that construct or the end of the
  # content.
  #
  # We don't currently need numbers so the tokenizer here just does sequences of
  # digits as a convenience to shrink the total number of tokens. If we needed numbers
  # later we would need a special handler for these much like strings.

  def start_multi_line_comment(s, i):
    return s[i] == '/' and i < len(s) - 1 and s[i + 1] == '*'

  def end_multi_line_comment(s, i):
    return s[i - 2] == '*' and s[i - 1] == '/'

  def start_single_line_comment(s, i):
    return s[i] == '-' and i < len(s) - 1 and s[i + 1] == '-'

  def end_single_line_comment(s, i):
    return s[i - 1] == '\n'

  def start_whitespace(s, i):
    return s[i].isspace()

  def end_whitespace(s, i):
    return not s[i].isspace()

  def start_number(s, i):
    return s[i].isdigit()

  def end_number(s, i):
    return not s[i].isdigit()

  def start_identifier(s, i):
    return s[i].isalpha() or s[i] == '_' or s[i] == '$'

  def end_identifier(s, i):
    return not(s[i].isalnum() or s[i] == '_')

  def start_string(s, i):
    return s[i] == '"' or s[i] == "'"

  def always_true(s, i):
    return True

  while i < len(sql):
    start = i
    if start_multi_line_comment(sql, i):
      i += 1
      end_checker = end_multi_line_comment
    elif start_single_line_comment(sql, i):
      i += 1
      end_checker = end_single_line_comment
    elif start_whitespace(sql, i):
      end_checker = end_whitespace
    elif start_identifier(sql, i):
      end_checker = end_identifier
    elif start_number(sql, i):
      end_checker = end_number
    elif start_string(sql, i):
      # Special handling here as we need to check for escaped closing quotes.
      quote = sql[i]
      end_checker = always_true
      i += 1
      while i < len(sql) and sql[i] != quote:
        i += 2 if sql[i] == '\\' else 1
    else:
      # We return single characters for everything else
      end_checker = always_true

    i += 1
    while i < len(sql) and not end_checker(sql, i):
      i += 1

    (yield sql[start:i])


def tokenize(sql):
  """ This is a basic tokenizer for our limited purposes.
  It splits a SQL statement up into a series of segments, where a segment is one of:
  - identifiers
  - left or right parentheses
  - multi-line comments
  - single line comments
  - white-space sequences
  - string literals
  - consecutive strings of characters that are not one of the items above

  The aim is for us to be able to find function calls (identifiers followed by '('), and the
  associated closing ')') so we can augment these if needed.

  Args:
    sql: a SQL statement as a (possibly multi-line) string.

  Returns:
    A list of strings corresponding to the groups above.
  """
  return list(_next_token(sql))
