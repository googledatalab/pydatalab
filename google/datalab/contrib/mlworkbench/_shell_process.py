# Copyright 2017 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Platform library - ml cell magic."""
from __future__ import absolute_import
from __future__ import unicode_literals


import os
import psutil
import six
import subprocess
import sys


def _wait_and_kill(pid_to_wait, pids_to_kill):
  """ Wait for a process to finish if it exists, and then kill a list of processes.

  Args:
    pid_to_wait: the process to wait for.
    pids_to_kill: a list of processes to kill after the process of pid_to_wait finishes.
  """
  if psutil.pid_exists(pid_to_wait):
    psutil.Process(pid=pid_to_wait).wait()

  for pid_to_kill in pids_to_kill:
    if psutil.pid_exists(pid_to_kill):
      p = psutil.Process(pid=pid_to_kill)
      p.kill()


def run_and_monitor(args, pid_to_wait, std_out_filter_fn=None, cwd=None):
  """ Start a process, and have it depend on another specified process.

  Args:
    args: the args of the process to start and monitor.
    pid_to_wait: the process to wait on. If the process ends, also kill the started process.
    std_out_filter_fn: a filter function which takes a string content from the stdout of the
        started process, and returns True if the string should be redirected to console stdout.
    cwd: the current working directory for the process to start.
  """

  monitor_process = None
  try:
    p = subprocess.Popen(args,
                         cwd=cwd,
                         env=os.environ,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)

    pids_to_kill = [p.pid]
    script = ('import %s;%s._wait_and_kill(%s, %s)' %
              (__name__, __name__, str(pid_to_wait), str(pids_to_kill)))
    monitor_process = subprocess.Popen(['python', '-c', script], env=os.environ)
    while p.poll() is None:
      line = p.stdout.readline()

      if not six.PY2:
        line = line.decode()

      if std_out_filter_fn is None or std_out_filter_fn(line):
        sys.stdout.write(line)
        # Cannot do sys.stdout.flush(). It appears that too many flush() calls will hang browser.
  finally:
    if monitor_process:
      monitor_process.kill()
