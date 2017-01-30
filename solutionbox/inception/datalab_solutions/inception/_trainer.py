# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Training implementation for inception model.
"""

import logging
import os
import tensorflow as tf
import time

from . import _util


def start_server(cluster, task):
  if not task.type:
    raise ValueError('--task_type must be specified.')
  if task.index is None:
    raise ValueError('--task_index must be specified.')

  # Create and start a server.
  return tf.train.Server(
      tf.train.ClusterSpec(cluster),
      protocol='grpc',
      job_name=task.type,
      task_index=task.index)

class Evaluator(object):
  """Loads variables from latest checkpoint and performs model evaluation."""

  def __init__(self, model, data_paths, batch_size, output_path, dataset='eval'):
    data_size = self._data_size(data_paths)
    if data_size <= batch_size:
      raise Exception('Data size is smaller than batch size.')
    self.num_eval_batches = data_size // batch_size
    self.batch_of_examples = []
    self.checkpoint_path = os.path.join(output_path, 'train')
    self.output_path = os.path.join(output_path, dataset)
    self.eval_data_paths = data_paths
    self.batch_size = batch_size
    self.model = model


  def _data_size(self, data_paths):
    n = 0
    options = tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    for file in data_paths:
      for line in tf.python_io.tf_record_iterator(file, options=options):
        n += 1
    return n

  def evaluate(self, num_eval_batches=None):
    """Run one round of evaluation, return loss and accuracy."""

    num_eval_batches = num_eval_batches or self.num_eval_batches
    with tf.Graph().as_default() as graph:
      self.tensors = self.model.build_eval_graph(self.eval_data_paths,
                                                 self.batch_size)
      self.summary = tf.merge_all_summaries()
      self.saver = tf.train.Saver()

    self.summary_writer = tf.train.SummaryWriter(self.output_path)
    self.sv = tf.train.Supervisor(
        graph=graph,
        logdir=self.output_path,
        summary_op=None,
        global_step=None,
        saver=self.saver)

    last_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
    with self.sv.managed_session(
        master='', start_standard_services=False) as session:
      self.sv.saver.restore(session, last_checkpoint)

      if not self.batch_of_examples:
        self.sv.start_queue_runners(session)
        for i in range(num_eval_batches):
          self.batch_of_examples.append(session.run(self.tensors.examples))

      for i in range(num_eval_batches):
        session.run(self.tensors.metric_updates,
                    {self.tensors.examples: self.batch_of_examples[i]})

      metric_values = session.run(self.tensors.metric_values)
      global_step = tf.train.global_step(session, self.tensors.global_step)
      summary = session.run(self.summary)
      self.summary_writer.add_summary(summary, global_step)
      self.summary_writer.flush()
      return metric_values



class Trainer(object):
  """Performs model training and optionally evaluation."""

  def __init__(self, input_dir, batch_size, max_steps, output_path, model, cluster, task):
    train_files, eval_files = _util.get_train_eval_files(input_dir)
    self.train_data_paths = train_files
    self.output_path = output_path
    self.batch_size = batch_size
    self.model = model
    self.max_steps = max_steps
    self.cluster = cluster
    self.task = task
    self.evaluator = Evaluator(self.model, eval_files, batch_size, output_path, 'eval_set')
    self.train_evaluator = Evaluator(self.model, train_files, batch_size, output_path, 'train_set')
    self.min_train_eval_rate = 20

  def run_training(self):
    """Runs a Master."""
    self.train_path = os.path.join(self.output_path, 'train')
    self.model_path = os.path.join(self.output_path, 'model')
    self.is_master = self.task.type != 'worker'
    log_interval = 15
    self.eval_interval = 30
    if self.is_master and self.task.index > 0:
      raise StandardError('Only one replica of master expected')

    if self.cluster:
      logging.info('Starting %s/%d', self.task.type, self.task.index)
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
          ps_device='/job:ps',
          worker_device='/job:%s/task:%d' % (self.task.type, self.task.index),
          cluster=self.cluster)
      # We use a device_filter to limit the communication between this job
      # and the parameter servers, i.e., there is no need to directly
      # communicate with the other workers; attempting to do so can result
      # in reliability problems.
      device_filters = [
          '/job:ps', '/job:%s/task:%d' % (self.task.type, self.task.index)
      ]
      config = tf.ConfigProto(device_filters=device_filters)
    else:
      target = ''
      device_fn = ''
      config = None

    with tf.Graph().as_default() as graph:
      with tf.device(device_fn):
        # Build the training graph.
        self.tensors = self.model.build_train_graph(self.train_data_paths,
                                                    self.batch_size)

        # Add the variable initializer Op.
        init_op = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver()

        # Build the summary operation based on the TF collection of Summaries.
        self.summary_op = tf.merge_all_summaries()

    # Create a "supervisor", which oversees the training process.
    self.sv = tf.train.Supervisor(
        graph,
        is_chief=self.is_master,
        logdir=self.train_path,
        init_op=init_op,
        saver=self.saver,
        # Write summary_ops by hand.
        summary_op=None,
        global_step=self.tensors.global_step,
        # No saving; we do it manually in order to easily evaluate immediately
        # afterwards.
        save_model_secs=0)

    should_retry = True
    to_run = [self.tensors.global_step, self.tensors.train]

    while should_retry:
      try:
        should_retry = False
        with self.sv.managed_session(target, config=config) as session:
          self.start_time = start_time = time.time()
          self.last_save = self.last_log = 0
          self.global_step = self.last_global_step = 0
          self.local_step = self.last_local_step = 0
          self.last_global_time = self.last_local_time = start_time

          # Loop until the supervisor shuts down or max_steps have
          # completed.
          max_steps = self.max_steps
          while not self.sv.should_stop() and self.global_step < max_steps:
            try:
              # Run one step of the model.
              self.global_step = session.run(to_run)[0]
              self.local_step += 1

              self.now = time.time()
              is_time_to_eval = (self.now - self.last_save) > self.eval_interval
              is_time_to_log = (self.now - self.last_log) > log_interval
              should_eval = self.is_master and is_time_to_eval
              should_log = is_time_to_log or should_eval

              if should_log:
                self.log(session)

              if should_eval:
                self.eval(session)
            except tf.errors.AbortedError:
              should_retry = True

          if self.is_master:
            # Take the final checkpoint and compute the final accuracy.
            # self.saver.save(session, self.sv.save_path, self.tensors.global_step)
            self.eval(session)

      except tf.errors.AbortedError:
        print('Hitting an AbortedError. Trying it again.')
        should_retry = True

    # Export the model for inference.
    if self.is_master:
      self.model.export(tf.train.latest_checkpoint(self.train_path), self.model_path)

    # Ask for all the services to stop.
    self.sv.stop()

  def log(self, session):
    """Logs training progress."""
    logging.info('Train [%s/%d], step %d (%.3f sec) %.1f '
                 'global steps/s, %.1f local steps/s', self.task.type,
                 self.task.index, self.global_step,
                 (self.now - self.start_time),
                 (self.global_step - self.last_global_step) /
                 (self.now - self.last_global_time),
                 (self.local_step - self.last_local_step) /
                 (self.now - self.last_local_time))
    self.last_log = self.now
    self.last_global_step, self.last_global_time = self.global_step, self.now
    self.last_local_step, self.last_local_time = self.local_step, self.now

  def eval(self, session):
    """Runs evaluation loop."""
    eval_start = time.time()
    self.saver.save(session, self.sv.save_path, self.tensors.global_step)
    logging.info(
        'Eval, step %d:\n- on train set %s\n-- on eval set %s',
        self.global_step,
        self.model.format_metric_values(self.train_evaluator.evaluate()),
        self.model.format_metric_values(self.evaluator.evaluate()))
    now = time.time()

    # Make sure eval doesn't consume too much of total time.
    eval_time = now - eval_start
    train_eval_rate = self.eval_interval / eval_time
    if train_eval_rate < self.min_train_eval_rate and self.last_save > 0:
      logging.info('Adjusting eval interval from %.2fs to %.2fs',
                   self.eval_interval, self.min_train_eval_rate * eval_time)
      self.eval_interval = self.min_train_eval_rate * eval_time

    self.last_save = now
    self.last_log = now

  def save_summaries(self, session):
    self.sv.summary_computed(session,
                             session.run(self.summary_op), self.global_step)
    self.sv.summary_writer.flush()

