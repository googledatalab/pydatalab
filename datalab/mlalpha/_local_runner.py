# Copyright 2016 Google Inc. All rights reserved.
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


import io
import json
import os
import psutil
import subprocess
import tempfile
import time

import google.cloud.ml as ml

import datalab.utils


def _wait_and_kill(pid_to_wait, pids_to_kill):
  """ Helper function.
      Wait for a process to finish if it exists, and then try to kill a list of processes.
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
      p.wait()


class LocalRunner(object):
  """Provides a "distributed" local run of a CloudML trainer packaged in a tarball.
     It simulates CloudML service by spawning master, worker, and ps processses,
     but instead of running in their own GKE cluster pods it runs all these as local processes.
  """

  def __init__(self, tar_path, module_to_run, logdir, replica_spec, program_args, all_args):
    """Initializes an instance of a LocalRunner

    Args:
      tar_path: the path of the trainer packaged in a tarball. Can be a local path or a GCS path.
      module_to_run: the module to run in the tarball.
      logdir: the directory to save logs.
      replica_spec: the number of replicas for each job_type.
          For example, {'master': 1, 'worker': 1, 'ps': 1}.
          Currently it supports at most one process for each job_type, and 'master' is required.
      program_args: the arguments of the training job program. For example,
          {
            'train_data_paths': ['/content/mydata/features_train'],
            'eval_data_paths': ['/content/mydata/features_eval'],
            'metadata_path': '/content/mydata/metadata.yaml',
            'output_path': '/content/mymodel/',
          }
      all_args: all args that can be submitted to cloud training such as job name, replicas, etc.
          It is aligned to the CloudML training service interface. In the program, it can be
          retrieved by 'TF_CONFIG' env var (json serialized) under 'job' member.

    Raises:
      Exception if replica_spec does not contain 'master' or its value is below one.
    """
    self._tar_path = tar_path
    self._module_to_run = module_to_run
    self._logdir = logdir
    self._replica_spec = replica_spec
    self._program_args = program_args
    if self._program_args is None:
      self._program_args = {}
    self._all_args = all_args
    if self._all_args is None:
      self._all_args = {}
    self._cluster_spec = self._create_cluster_spec()
    self._task_processes = {}
    self._log_writers = {}
    self._log_readers = {}
    self._monitor_process = None

  def _create_cluster_spec(self):
    """Create cluster spec that will be passed to each task process as command parameter.
       This matches CloudML training service behavior.
    """
    spec = {}
    for job_type, replicas in self._replica_spec.iteritems():
      if replicas > 0:
        port = datalab.utils.pick_unused_port()
        spec[job_type] = ['localhost:' + str(port)]
    if 'master' not in spec:
      raise Exception('Need to have at least 1 master replica')
    return spec

  def _create_tf_config(self, job_type):
    """Create a list of arguments that will be passed to task process as command
       parameters. This matches CloudML training service behavior.
    """
    task_spec = {'type': job_type, 'index': 0}
    return {
      'cluster': self._cluster_spec,
      'task': task_spec,
      'job': self._all_args,
    }

  def _create_task_args(self):
    args = [
      'python',
      '-m',
      self._module_to_run,
    ]
    for k,v in self._program_args.iteritems():
      if isinstance(v, list):
        for item in v:
          args.append('--' + k)
          args.append(str(item))
      else:
        args.append('--' + k)
        args.append(str(v))
    return args

  def _extract_tar(self):
    extracted_dir = tempfile.mkdtemp()
    tar_path = self._tar_path
    if tar_path.startswith('gs://'):
      tar_path = os.path.join(extracted_dir, os.path.basename(tar_path))
      ml.util._file.copy_file(self._tar_path, tar_path)
    subprocess.check_call(['pip', 'install', tar_path, '--target', extracted_dir,
                           '--upgrade', '--force-reinstall'])
    return extracted_dir

  def _clean_up(self):
    processes_to_clean = list(self._task_processes.values())
    if self._monitor_process is not None:
      processes_to_clean.append(self._monitor_process)

    for p in processes_to_clean:
      if p.poll() is None:
        # TODO(qimingj): consider p.kill() if it does not terminate in certain time.
        p.terminate()
        p.wait()

    for k,v in self._log_readers.iteritems():
      v.close()
    for k,v in self._log_writers.iteritems():
      v.close()

  def _start_externel_monitor_process(self):
    """Create a process that monitors the current process. If the current process exists,
       Clean up a list of target processes.
       This is needed to kill all running training processes when the kernel is restarted.
       Note that Jupyter does not kill child processes when kernel is restarted. "atexit"
       hook doesn't work either if the kernel is busy such as in time.sleep (seems SIGKILL
       is used to restart the kernel).
    """
    pids_to_kill = [p.pid for p in self._task_processes.values()]
    script = 'import %s; %s._wait_and_kill(%s, %s)' % \
        (__name__, __name__, str(os.getpid()), str(pids_to_kill))
    self._monitor_process = subprocess.Popen(['python', '-c', script])

  def _start_task_process(self, workdir, job_type):
    args = self._create_task_args()
    logfile = os.path.join(self._logdir, job_type)
    # We cannot pipe child process's stdout and stderr directly because
    # we need to append 'master', 'worker', 'ps' at begining of each
    # log entry. A memory stream such as StringIO does not work here because
    # Popen expects file descriptor. Therefore physical files are used to back
    # up the messages.
    w = io.open(logfile, 'w')
    r = io.open(logfile, 'r')
    tf_config = self._create_tf_config(job_type)
    env = os.environ.copy()
    env['TF_CONFIG'] = json.dumps(tf_config)
    p = subprocess.Popen(args, env=env, stdout=w, stderr=w)
    self._log_writers[job_type] = w
    self._log_readers[job_type] = r
    self._task_processes[job_type] = p

  def _print_task_output(self, callback, param, done):
    if callback is None:
      return
    new_msgs = []
    for job_type, reader in self._log_readers.iteritems():
      content = reader.read()
      if (content):
        lines = content.split('\n')
        for line in lines:
          new_msgs.append(job_type + ': '+ line)
    callback(self._replica_spec, new_msgs, done, param)

  def run(self, callback, param, interval):
    """Run a training job locally. Block the caller until it finishes.
       Prints out task processes stdout and stderr.

    Args:
      callback: a callback that will be invoked every "interval" seconds. The signature is:
          callback(replica_spec, new_msgs, done, param), where:
              replica_spec: the replica spec of the runner
              new_msgs: new output messages that are available from all task processes
              done: whether the job is finished
              param: a callback param.
      param: the callback param that will be passed along whenever the callback is invoked.
      interval: the interval in seconds controlling the callback frequency.
    """
    workdir = self._extract_tar()
    previous_cwd = os.getcwd()
    os.chdir(workdir)
    for job_type, replicas in self._replica_spec.iteritems():
      if replicas > 0:
        self._start_task_process(workdir, job_type)
    os.chdir(previous_cwd)
    self._start_externel_monitor_process()
    while self._task_processes['master'].poll() is None:
      self._print_task_output(callback, param, False)
      time.sleep(interval)
    self._print_task_output(callback, param, True)
    self._clean_up()
