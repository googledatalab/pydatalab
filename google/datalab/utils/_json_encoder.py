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

""" JSON encoder that can handle Python datetime objects. """

from __future__ import absolute_import
from __future__ import unicode_literals
import datetime
import json


class JSONEncoder(json.JSONEncoder):
  """ A JSON encoder that can handle Python datetime objects. """

  def default(self, obj):
    if isinstance(obj, datetime.date) or isinstance(obj, datetime.datetime):
      return obj.isoformat()
    elif isinstance(obj, datetime.timedelta):
      return (datetime.datetime.min + obj).time().isoformat()
    else:
      return super(JSONEncoder, self).default(obj)
