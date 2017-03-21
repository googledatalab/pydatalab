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

"""Implements Job functionality for async tasks."""
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import str
from builtins import object

import concurrent.futures
import datetime
import time
import traceback
import uuid


class JobError(Exception):
  """ A helper class to capture multiple components of Job errors.  """

  def __init__(self, location, message, reason):
    self.location = location
    self.message = message
    self.reason = reason

  def __str__(self):
    return '%s %s %s' % (self.message, self.reason, self.location)


class Job(object):
  """A manager object for async operations.

     A Job can have a Future in which case it will be able to monitor its own completion state
     and result, or it may have no Future in which case it must be a derived class that
     manages this some other way. We do this instead of having an abstract base class in
     order to make wait_one/wait_all more efficient; instead of just sleeping and polling
     we can use more reactive ways of monitoring groups of Jobs.
  """

  _POLL_INTERVAL_SECONDS = 5

  def __init__(self, job_id=None, future=None):
    """Initializes an instance of a Job.

    Args:
      job_id: a unique ID for the job. If None, a UUID will be generated.
      future: the Future associated with the Job, if any.
    """
    self._job_id = str(uuid.uuid4()) if job_id is None else job_id
    self._future = future
    self._is_complete = False
    self._errors = None
    self._fatal_error = None
    self._result = None
    self._start_time = datetime.datetime.utcnow()
    self._end_time = None

  def __str__(self):
    return self._job_id

  @property
  def id(self):
    """ Get the Job ID.

    Returns:
      The ID of the job.
    """
    return self._job_id

  @property
  def is_complete(self):
    """ Get the completion state of the job.

    Returns:
      True if the job is complete; False if it is still running.
    """
    self._refresh_state()
    return self._is_complete

  @property
  def failed(self):
    """ Get the success state of the job.

    Returns:
      True if the job failed; False if it is still running or succeeded (possibly with partial
      failure).
    """
    self._refresh_state()
    return self._is_complete and self._fatal_error is not None

  @property
  def fatal_error(self):
    """ Get the job error.

    Returns:
      None if the job succeeded or is still running, else the error tuple for the failure.
    """
    self._refresh_state()
    return self._fatal_error

  @property
  def errors(self):
    """ Get the non-fatal errors in the job.

    Returns:
      None if the job is still running, else the list of errors that occurred.
    """
    self._refresh_state()
    return self._errors

  def result(self):
    """ Get the result for a job. This will block if the job is incomplete.

    Returns:
      The result for the Job.

    Raises:
      An exception if the Job resulted in an exception.

    """
    self.wait()
    if self._fatal_error:
      raise self._fatal_error
    return self._result

  @property
  def start_time_utc(self):
    """ The UTC start time of the job as a Python datetime. """
    return self._start_time

  @property
  def end_time_utc(self):
    """ The UTC end time of the job (or None if incomplete) as a Python datetime. """
    return self._end_time

  @property
  def total_time(self):
    """ The total time in fractional seconds that the job took, or None if not complete. """
    if self._end_time is None:
      return None
    return (self._end_time - self._start_time).total_seconds()

  def _refresh_state(self):
    """ Get the state of a job. Must be overridden by derived Job classes
        for Jobs that don't use a Future.
    """
    if self._is_complete:
      return

    if not self._future:
      raise Exception('Please implement this in the derived class')

    if self._future.done():
      self._is_complete = True
      self._end_time = datetime.datetime.utcnow()
      try:
        self._result = self._future.result()
      except Exception as e:
        message = str(e)
        self._fatal_error = JobError(location=traceback.format_exc(), message=message,
                                     reason=str(type(e)))

  def _timeout(self):
    """ Helper for raising timeout errors. """
    raise concurrent.futures.TimeoutError('Timed out waiting for Job %s to complete' % self._job_id)

  def wait(self, timeout=None):
    """ Wait for the job to complete, or a timeout to happen.

    Args:
      timeout: how long to wait before giving up (in seconds); default None which means no timeout.

    Returns:
      The Job
    """
    if self._future:
      try:
        # Future.exception() will return rather than raise any exception so we use it.
        self._future.exception(timeout)
      except concurrent.futures.TimeoutError:
        self._timeout()
      self._refresh_state()
    else:
      # fall back to polling
      while not self.is_complete:
        if timeout is not None:
          if timeout <= 0:
            self._timeout()
          timeout -= Job._POLL_INTERVAL_SECONDS
        time.sleep(Job._POLL_INTERVAL_SECONDS)
    return self

  @property
  def state(self):
    """ Describe the state of a Job.

    Returns: A string describing the job's state.
    """
    state = 'in progress'
    if self.is_complete:
      if self.failed:
        state = 'failed with error: %s' % str(self._fatal_error)
      elif self._errors:
        state = 'completed with some non-fatal errors'
      else:
        state = 'completed'
    return state

  def __repr__(self):
    """ Get the notebook representation for the job. """
    return 'Job %s %s' % (self._job_id, self.state)

  @staticmethod
  def _wait(jobs, timeout, return_when):
    # If a single job is passed in, make it an array for consistency
    if isinstance(jobs, Job):
      jobs = [jobs]
    elif len(jobs) == 0:
      return jobs

    wait_on_one = return_when == concurrent.futures.FIRST_COMPLETED
    completed = []
    while True:
      if timeout is not None:
        timeout -= Job._POLL_INTERVAL_SECONDS

      done = [job for job in jobs if job.is_complete]

      if len(done):
        completed.extend(done)
        for job in done:
          jobs.remove(job)
        if wait_on_one or len(jobs) == 0:
          return completed

      if timeout is not None and timeout < 0:
        return completed

      # Need to block for some time. Favor using concurrent.futures.wait if possible
      # as it can return early if a (thread) job is ready; else fall back to time.sleep.
      futures = [job._future for job in jobs if job._future]
      if len(futures) == 0:
        time.sleep(Job._POLL_INTERVAL_SECONDS)
      else:
        concurrent.futures.wait(futures, timeout=Job._POLL_INTERVAL_SECONDS,
                                return_when=return_when)
