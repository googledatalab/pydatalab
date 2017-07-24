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

"""Test MLWorkbench shell process monitor"""
from __future__ import absolute_import
from __future__ import print_function

import logging
import os
import psutil
import six
import subprocess
import sys
import threading
import time
import unittest

import google.datalab.contrib.mlworkbench._shell_process as _shell_process


class TestShellProcess(unittest.TestCase):
  """Tests for process managements used in MLWorkbench magics."""

  def setUp(self):
    self._logger = logging.getLogger('TestStructuredDataLogger')
    self._logger.setLevel(logging.DEBUG)
    if not self._logger.handlers:
      self._logger.addHandler(logging.StreamHandler(stream=sys.stdout))

  def test_process(self):
    """ Test starting a process and have it depend on another process.

    Steps:
      1: process_to_wait is created with 10 seconds to live. It is assumed this
         test will finish before then.
      2: _shell_process.run_and_monitor() creates a worker process and a monitor
         process.
      3: The worker process prints "exclude message", "include message1".
      4: This test wakes up from 1 sec sleep, kills process_to_wait.
      5: The monitor process seeks process_to_wait is dead, and kills the worker.
         Monitor process also ends after killing the worker.
      6: This test wakes up from 1 sec sleep and checks the worker is dead. Also
         checks the stdout message of the worker process is captured and filtered.
    """

    py_cmd = 'python3' if six.PY3 else 'python'
    # The process will do time.sleep(10) but it will be killed much earlier.
    process_to_wait_args = [py_cmd, '-c', 'import time; time.sleep(10)']
    process_to_wait = subprocess.Popen(process_to_wait_args, env=os.environ)
    self.assertIsNone(process_to_wait.poll())
    self._logger.debug('TestProcess: Started a process %d which will be waited on.' %
                       process_to_wait.pid)

    # The worker process prints out 3 messages. "exclude message" should be filtered out.
    # "include message1" should be redirected to stdout. "include message2" should not
    # get a chance to output because the worker process should have been killed by
    # monitor process by then.
    worker_process_args = [
        py_cmd,
        '-c',
        'import time;import sys;print(\'exclude message\');print(\'include message1\');' +
        'sys.stdout.flush();time.sleep(5);print(\'include message2\');sys.stdout.flush()'
    ]

    old_stdout = sys.stdout
    buf = six.StringIO()
    sys.stdout = buf
    t = threading.Thread(target=_shell_process.run_and_monitor,
                         args=(worker_process_args,
                               process_to_wait.pid, lambda x: x.startswith('include')))
    t.start()
    time.sleep(1)

    # The worker process should have been started by _shell_process.run_and_monitor.
    # But because the process id is not available outside the function, we need to search
    # all processes to find the one that has matching args.
    process_to_monitor = None
    for proc in psutil.process_iter():
      if proc.is_running() and proc.cmdline() == worker_process_args:
        process_to_monitor = proc
        break

    self.assertIsNotNone(process_to_monitor)
    self._logger.debug('TestProcess: Started a worker process %d to be monitored on.' %
                       process_to_monitor.pid)

    self._logger.debug('TestProcess: killing process %d' % process_to_wait.pid)
    process_to_wait.kill()
    process_to_wait.wait()

    # Wait for 3 sec for the process_to_monitor to be killed.
    for i in range(3):
      time.sleep(1)
      if not process_to_monitor.is_running():
        break

    self.assertFalse(process_to_monitor.is_running())
    self._logger.debug('TestProcess: process %d being monitored ' % process_to_monitor.pid +
                       'was also killed successfully after waiting process was killed.')
    sys.stdout = old_stdout

    # Make sure stdout filter works.
    self.assertEqual('include message1', buf.getvalue().rstrip())


if __name__ == '__main__':
    unittest.main()
