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

from __future__ import absolute_import
from __future__ import unicode_literals
import os
import sys
import unittest

# Set up the path so that we can import our google.datalab.* packages.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))  # noqa

import bigquery.api_tests
import bigquery.dataset_tests
import bigquery.external_data_source_tests
import bigquery.jobs_tests
import bigquery.parser_tests
import bigquery.query_tests
import bigquery.sampling_tests
import bigquery.schema_tests
import bigquery.table_tests
# import bigquery.udf_tests
import bigquery.view_tests
import kernel.bigquery_tests
import kernel.chart_data_tests
import kernel.chart_tests
import kernel.commands_tests
import kernel.html_tests
import kernel.storage_tests
import kernel.utils_tests
import mltoolbox_structured_data.traininglib_tests
import mltoolbox_structured_data.dl_interface_tests
import stackdriver.commands.monitoring_tests
import stackdriver.monitoring.group_tests
import stackdriver.monitoring.metric_tests
import stackdriver.monitoring.resource_tests
import stackdriver.monitoring.query_metadata_tests
import stackdriver.monitoring.query_tests
import stackdriver.monitoring.utils_tests
import storage.api_tests
import storage.bucket_tests
import storage.object_tests
import _util.http_tests
import _util.lru_cache_tests
import _util.util_tests


_TEST_MODULES = [
    bigquery.api_tests,
    bigquery.dataset_tests,
    # bigquery.external_data_source_tests, # TODO: enable external data source tests
    bigquery.jobs_tests,
    bigquery.parser_tests,
    bigquery.query_tests,
    bigquery.sampling_tests,
    bigquery.schema_tests,
    bigquery.table_tests,
    # bigquery.udf_tests, # TODO: enable UDF tests once new implementation is done
    bigquery.view_tests,
    bigquery.sampling_tests,
    kernel.bigquery_tests,
    kernel.chart_data_tests,
    kernel.chart_tests,
    kernel.commands_tests,
    kernel.html_tests,
    kernel.storage_tests,
    kernel.utils_tests,
    mltoolbox_structured_data.dl_interface_tests,
    stackdriver.commands.monitoring_tests,
    stackdriver.monitoring.group_tests,
    stackdriver.monitoring.metric_tests,
    stackdriver.monitoring.resource_tests,
    stackdriver.monitoring.query_metadata_tests,
    stackdriver.monitoring.query_tests,
    stackdriver.monitoring.utils_tests,
    storage.api_tests,
    storage.bucket_tests,
    storage.object_tests,
    _util.http_tests,
    _util.lru_cache_tests,
    _util.util_tests
]

# mltoolbox is not part of the datalab install, but it should still be tested.
# mltoolbox does not work with python 3.
if sys.version_info.major == 2:
  _TEST_MODULES.append(mltoolbox_structured_data.traininglib_tests)

if __name__ == '__main__':
  suite = unittest.TestSuite()
  for m in _TEST_MODULES:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(m))

  runner = unittest.TextTestRunner()
  result = runner.run(suite)

  sys.exit(len(result.errors) + len(result.failures))
