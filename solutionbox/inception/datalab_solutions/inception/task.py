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


"""Entry point for CloudML training.

   CloudML training requires a tarball package and a python module to run. This file
   provides such a "main" method and a list of args passed with the program.
"""

import argparse
import json
import logging
import os
import tensorflow as tf

from . import _model
from . import _trainer
from . import _util


def main(_):
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_data_paths',
      type=str,
      action='append',
      help='The paths to the training data files. '
      'Can be comma separated list of files or glob pattern.')
  parser.add_argument(
      '--input_dir',
      type=str,
      help='The input dir path for training and evaluation data.')
  parser.add_argument(
      '--output_path',
      type=str,
      help='The path to which checkpoints and other outputs '
      'should be saved. This can be either a local or GCS '
      'path.')
  parser.add_argument(
      '--max_steps',
      type=int,)
  parser.add_argument(
      '--batch_size',
      type=int,
      help='Number of examples to be processed per mini-batch.')
  parser.add_argument(
      '--num_classes',
      type=int,
      help='Number of classification classes.')
  parser.add_argument(
      '--checkpoint',
      type=str,
      default=_util._DEFAULT_CHECKPOINT_GSURL,
      help='Pretrained inception checkpoint path.')

  args, _ = parser.parse_known_args()
  model = _model.Model(args.num_classes, 0.5, args.checkpoint)

  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  # Print the job data as provided by the service.
  logging.info('Original job data: %s', env.get('job', {}))
  task_data = env.get('task', None) or {'type': 'master', 'index': 0}
  task = type('TaskSpec', (object,), task_data)
  trial = task_data.get('trial')
  if trial is not None:
    args.output_path = os.path.join(args.output_path, trial)

  cluster_data = env.get('cluster', None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
  if not cluster or not task or task.type == 'master' or task.type == 'worker':
     _trainer.Trainer(args.input_dir, args.batch_size, args.max_steps,
                      args.output_path, model, cluster, task).run_training()
  elif task.type == 'ps':
     server = _trainer.start_server(cluster, task)
     server.join()
  else:
    raise ValueError('invalid task_type %s' % (task.type,))

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
