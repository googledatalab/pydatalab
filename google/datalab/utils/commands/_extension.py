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

"""Google Cloud Platform library - Extension cell magic."""
from __future__ import absolute_import
from __future__ import unicode_literals

try:
  import IPython
  import IPython.core.display
  import IPython.core.magic
except ImportError:
  raise Exception('This module can only be loaded in ipython.')

from . import _commands
from . import _utils


@IPython.core.magic.register_line_cell_magic
def extension(line, cell=None):
  """ Load an extension. Use %extension --help for more details. """
  parser = _commands.CommandParser(prog='%extension', description="""
Load an extension into Datalab. Currently only mathjax is supported.
""")
  subparser = parser.subcommand('mathjax', 'Enabled MathJaX support in Datalab.')
  subparser.set_defaults(ext='mathjax')
  parser.set_defaults(func=_extension)
  return _utils.handle_magic_line(line, cell, parser)


def _extension(args, cell):
  ext = args['ext']
  if ext == 'mathjax':
    # TODO: remove this with the next version update
    # MathJax is now loaded by default for all notebooks
    return
  raise Exception('Unsupported extension %s' % ext)

