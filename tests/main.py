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
import argparse
import sys
import unittest

# For these tests to work locally, install the package with "pip install -e ."
# from the parent folder and run "python tests/main.py"

import context_tests
import bigquery.api_tests
import bigquery.dataset_tests
import bigquery.external_data_source_tests
import bigquery.jobs_tests
import bigquery.parser_tests
import bigquery.query_tests
import bigquery.sampling_tests
import bigquery.schema_tests
import bigquery.table_tests
import bigquery.udf_tests
import bigquery.view_tests
import kernel.bigquery_tests
import kernel.chart_data_tests
import kernel.chart_tests
import kernel.html_tests
import kernel.pipeline_tests
import kernel.storage_tests
import kernel.utils_tests
import ml.confusion_matrix_tests
import ml.dataset_tests
import ml.facets_tests
import ml.summary_tests
import ml.tensorboard_tests
import mltoolbox_code_free_ml.all_tests
import mltoolbox_structured_data.dl_interface_tests
import mltoolbox_structured_data.sd_e2e_tests
import mltoolbox_structured_data.traininglib_tests
import mlworkbench_magic.archive_tests
import mlworkbench_magic.explainer_tests
import mlworkbench_magic.local_predict_tests
import mlworkbench_magic.ml_tests
import mlworkbench_magic.shell_process_tests
import pipeline.pipeline_tests
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
import _util.commands_tests
import _util.feature_statistics_generator_test
import _util.generic_feature_statistics_generator_test
import _util.http_tests
import _util.lru_cache_tests
import _util.util_tests


_UNIT_TEST_MODULES = [
    context_tests,
    bigquery.api_tests,
    bigquery.dataset_tests,
    # bigquery.external_data_source_tests, # TODO: enable external data source tests
    bigquery.jobs_tests,
    bigquery.parser_tests,
    bigquery.query_tests,
    bigquery.sampling_tests,
    bigquery.schema_tests,
    bigquery.table_tests,
    bigquery.udf_tests,
    bigquery.view_tests,
    bigquery.sampling_tests,
    kernel.bigquery_tests,
    kernel.chart_data_tests,
    kernel.chart_tests,
    kernel.html_tests,
    kernel.pipeline_tests,
    kernel.storage_tests,
    kernel.utils_tests,
    ml.confusion_matrix_tests,
    ml.dataset_tests,
    ml.facets_tests,
    ml.summary_tests,
    mlworkbench_magic.ml_tests,
    pipeline.pipeline_tests,
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
    _util.commands_tests,
    _util.feature_statistics_generator_test,
    _util.generic_feature_statistics_generator_test,
    _util.http_tests,
    _util.lru_cache_tests,
    _util.util_tests
]


_INTEGRATION_TEST_MODULES = [
    ml.tensorboard_tests,
    mltoolbox_code_free_ml.all_tests,
    mltoolbox_structured_data.dl_interface_tests,
    mltoolbox_structured_data.sd_e2e_tests,  # Not everything runs in Python 3.
    mltoolbox_structured_data.traininglib_tests,
    mlworkbench_magic.local_predict_tests,
    mlworkbench_magic.explainer_tests,
    # TODO: the test fails in travis only. Need to investigate.
    # mlworkbench_magic.shell_process_tests,
    mlworkbench_magic.archive_tests,
]


def parse_arguments(argv):
  """Parse command line arguments.

  Args:
    argv: list of command line arguments, including program name.

  Returns:
    An argparse Namespace object.

  Raises:
    ValueError: for bad parameters
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--unittestonly',
                      action='store_true',
                      help='Only run unit tests (no integration tests).')
  args = parser.parse_args(args=argv[1:])
  return args


if __name__ == '__main__':
  args = parse_arguments(sys.argv)
  suite = unittest.TestSuite()
  for m in _UNIT_TEST_MODULES:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(m))
  if not args.unittestonly:
    for m in _INTEGRATION_TEST_MODULES:
      suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(m))

  runner = unittest.TextTestRunner()
  result = runner.run(suite)

  sys.exit(len(result.errors) + len(result.failures))
