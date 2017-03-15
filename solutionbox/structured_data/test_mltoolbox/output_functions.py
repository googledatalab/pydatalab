# Copyright 2017 Google Inc. All rights reserved.
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
from __future__ import absolute_import
from __future__ import unicode_literals

import sys

from contextlib import contextmanager
from StringIO import StringIO


@contextmanager
def captured_output(silent_output=False):
  """Allows redirecting stdour and stderr to a string buffer.

  Usage:
    with captured_output() as (stdout_str, stderr_str):
      print('hello world')


  Args:
    silent_output: if Ture, output is redirected. If false, outpout is not 
        redirected.
  """
  if silent_output:
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err
  else:
    yield sys.stdout, sys.stderr
