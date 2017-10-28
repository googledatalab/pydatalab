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


try:
  import IPython
except ImportError:
  raise Exception('This module can only be loaded in ipython.')

import argparse
import os
import pandas as pd
import psutil
import subprocess
import time

import google.datalab as datalab


class TensorBoard(object):
  """Start, shutdown, and list TensorBoard instances."""

  @staticmethod
  def list():
    """List running TensorBoard instances."""

    running_list = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir')
    parser.add_argument('--port')
    for p in psutil.process_iter():
      if p.name() != 'tensorboard' or p.status() == psutil.STATUS_ZOMBIE:
        continue
      cmd_args = p.cmdline()
      del cmd_args[0:2]  # remove 'python' and 'tensorboard'
      args = parser.parse_args(cmd_args)
      running_list.append({'pid': p.pid, 'logdir': args.logdir, 'port': args.port})
    return pd.DataFrame(running_list)

  @staticmethod
  def start(logdir):
    """Start a TensorBoard instance.

    Args:
      logdir: the logdir to run TensorBoard on.
    Raises:
      Exception if the instance cannot be started.
    """
    if logdir.startswith('gs://'):
      # Check user does have access. TensorBoard will start successfully regardless
      # the user has read permissions or not so we check permissions here to
      # give user alerts if needed.
      datalab.storage._api.Api.verify_permitted_to_read(logdir)

    port = datalab.utils.pick_unused_port()
    args = ['tensorboard', '--logdir=' + logdir, '--port=' + str(port)]
    p = subprocess.Popen(args)
    retry = 10
    while (retry > 0):
      if datalab.utils.is_http_running_on(port):
        basepath = os.environ.get('DATALAB_ENDPOINT_URL', '')
        url = '%s/_proxy/%d/' % (basepath.rstrip('/'), port)
        html = '<p>TensorBoard was started successfully with pid %d. ' % p.pid
        html += 'Click <a href="%s" target="_blank">here</a> to access it.</p>' % url
        IPython.display.display_html(html, raw=True)
        return p.pid
      time.sleep(1)
      retry -= 1

    raise Exception('Cannot start TensorBoard.')

  @staticmethod
  def stop(pid):
    """Shut down a specific process.

    Args:
      pid: the pid of the process to shutdown.
    """
    if psutil.pid_exists(pid):
      try:
        p = psutil.Process(pid)
        p.kill()
      except Exception:
        pass
