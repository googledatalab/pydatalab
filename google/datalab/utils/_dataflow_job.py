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

"""Implements DataFlow Job functionality."""


from google.datalab import _job


class DataflowJob(_job.Job):
  """Represents a DataFlow Job.
  """

  def __init__(self, runner_results):
    """Initializes an instance of a DataFlow Job.

    Args:
      runner_results: a DataflowPipelineResult returned from Pipeline.run().
    """
    super(DataflowJob, self).__init__(runner_results._job.name)
    self._runner_results = runner_results

  def _refresh_state(self):
    """ Refresh the job info. """

    # DataFlow's DataflowPipelineResult does not refresh state, so we have to do it ourselves
    # as a workaround.
    self._runner_results._job = (
        self._runner_results._runner.dataflow_client.get_job(self._runner_results.job_id()))
    self._is_complete = self._runner_results.state in ['STOPPED', 'DONE', 'FAILED', 'CANCELLED']
    self._fatal_error = getattr(self._runner_results._runner, 'last_error_msg', None)
    # Sometimes Dataflow does not populate runner.last_error_msg even if the job fails.
    if self._fatal_error is None and self._runner_results.state == 'FAILED':
      self._fatal_error = 'FAILED'
