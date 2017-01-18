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


"""Provides interface for Datalab. It provides:
     local_preprocess
     local_train
     local_predict
     cloud_preprocess
     cloud_train
     cloud_predict
   Datalab will look for functions with the above names.
"""

import logging
import os
import urllib
import subprocess
import sys
import datetime

def _percent_flags(train_percent=None, eval_percent=None, test_percent=None):
  """Convert train/eval/test percents into command line flags.

  Args:
    train_percent: Int in range [0, 100].
    eval_percent: Int in range [0, 100].
    train_percent: Int in range [0, 100].

  Returns:
    Array of strings encoding the command line flags.
  """
  train_percent = train_percent or 0
  eval_percent = eval_percent or 0
  test_percent = test_percent or 0
  if train_percent == 0 and eval_percent == 0 and test_percent == 0:
    percent_flags = []
  else:
    percent_flags = ['--train_percent=%s' % str(train_percent),
                     '--eval_percent=%s' % str(eval_percent),
                     '--test_percent=%s' % str(test_percent)]
  return percent_flags


def _default_project():
  import datalab.context
  context = datalab.context.Context.default()
  return context.project_id

def _is_in_IPython():
  try:
    import IPython
    return True
  except ImportError:
    return False

def _check_transforms_config_file(transforms_config_file):
  """Check that the transforms file has expected values."""
  pass


def _run_cmd(cmd):
  output = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

  while True:
    line = output.stdout.readline().rstrip()
    print(line)
    if line == '' and output.poll() != None:
      break

def local_preprocess(input_file_path, output_dir, transforms_config_file,
                     train_percent=None, eval_percent=None, test_percent=None):
  """Preprocess data locally with Beam. 

  Produce output that can be used by training efficiently. Will also split 
  data into three sets (training, eval, and test). {train, eval, test}_percent
  should be nonnegative integers that sum to 100. 

  Args:
    input_file_path: String. File pattern what will expand into a list of csv 
        files. Preprocessing will automatically slip the data into three sets
        for training, evaluation, and testing. Can be local or GCS path. 
    output_dir: The output directory to use; can be local or GCS path. 
    transforms_config_file: File path to the config file.
    train_percent: Int in range [0, 100].
    eval_percent: Int in range [0, 100].
    train_percent: Int in range [0, 100].
  """
  _check_transforms_config_file(transforms_config_file)

  percent_flags = _percent_flags(train_percent, eval_percent, test_percent)
  this_folder = os.path.dirname(os.path.abspath(__file__))

  cmd = ['python',
         os.path.join(this_folder, 'preprocess/preprocess.py'),
         '--input_file_path=%s' % input_file_path,
         '--output_dir=%s' % output_dir,
         '--transforms_config_file=%s' % transforms_config_file] + percent_flags
  
  print('Local preprocess, running command: %s' % ' '.join(cmd))
  _run_cmd(' '.join(cmd))

  print 'Local preprocessing done.'


def cloud_preprocess(input_file_path, output_dir, transforms_config_file,
                     train_percent=None, eval_percent=None, test_percent=None,
                     project_id=None, job_name=None):
  """Preprocess data in the cloud with Dataflow. 

  Produce output that can be used by training efficiently. Will also split 
  data into three sets (training, eval, and test). {train, eval, test}_percent
  should be nonnegative integers that sum to 100. 

  Args:
    input_file_path: String. File pattern what will expand into a list of csv 
        files. Preprocessing will automatically slip the data into three sets
        for training, evaluation, and testing. Can be local or GCS path. 
    output_dir: The output directory to use; can be local or GCS path. 
    transforms_config_file: File path to the config file.
    train_percent: Int in range [0, 100].
    eval_percent: Int in range [0, 100].
    train_percent: Int in range [0, 100].
    project_id: String. The GCE project to use. Defaults to the notebook's
        default project id.
    job_name: String. Job name as listed on the Dataflow service. If None, a 
        default job name is selected.
  """
  _check_transforms_config_file(transforms_config_file)

  percent_flags = _percent_flags(train_percent, eval_percent, test_percent)
  this_folder = os.path.dirname(os.path.abspath(__file__))
  project_id = project_id or _default_project()
  job_name = job_name or ('structured-data-' + 
                          datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

  cmd = ['python',
         os.path.join(this_folder, 'preprocess/preprocess.py'),
         '--cloud',
         '--project_id=%s' % project_id,
         '--job_name=%s' % job_name,
         '--input_file_path=%s' % input_file_path,
         '--output_dir=%s' % output_dir,
         '--transforms_config_file=%s' % transforms_config_file] + percent_flags
  
  print('Cloud preprocess, running command: %s' % ' '.join(cmd))
  _run_cmd(' '.join(cmd))

  print 'Cloud preprocessing job submitted.'

  if (_is_in_IPython()):
    import IPython
    
    dataflow_url = ('https://console.developers.google.com/dataflow?project=%s' 
                    % project_id)
    html = ('<p>Click <a href="%s" target="_blank">here</a> to track '
            'preprocessing job.</p><br/>' % dataflow_url)
    IPython.display.display_html(html, raw=True)



def local_train(preprocessed_dir, transforms_config_file, output_dir,
                layer_sizes=None, max_steps=None):
  """Train model locally.
  Args:
    preprocessed_dir: The output directory from preprocessing. Must contain 
        files named features_train*.tfrecord.gz, features_eval*.tfrecord.gz,
        and metadata.json. Can be local or GCS path.
    transforms_config_file: File path to the config file.
    output_dir: Output directory of training.
    layer_sizes: String. Represents the layers in the connected DNN. 
        If the model type is DNN, this must be set. Example "10 3 2", this will
        create three DNN layers where the first layer will have 10 nodes, the
        middle layer will have 3 nodes, and the laster layer will have 2 nodes.
    max_steps: Int. Number of training steps to perform.
  """
  
  #TODO(brandondutra): fix to TF 0.12
  #TODO(brandondutra): allow other flags to be set.

  train_filename = os.path.join(preprocessed_dir, 'features_train*')
  eval_filename = os.path.join(preprocessed_dir, 'features_eval*')
  metadata_filename = os.path.join(preprocessed_dir, 'metadata.json')
  this_folder = os.path.dirname(os.path.abspath(__file__))
  cmd = ['cd %s &&' % this_folder, 'gcloud alpha ml local train',
         '--module-name=trainer.task',
         '--package-path=%s' % os.path.join(this_folder, 'trainer/task.py'),
         '--',
         '--train_data_paths=%s' % train_filename,
         '--eval_data_paths=%s' % eval_filename,
         '--metadata_path=%s' % metadata_filename,
         '--output_path=%s' % output_dir,
         '--transforms_config_file=%s' % transforms_config_file,
         '--max_steps=%s' % str(max_steps)]
  if layer_sizes:
    cmd += ['--layer_sizes %s' % layer_sizes]

  print('Local training, running command: %s' % ' '.join(cmd))
  _run_cmd(' '.join(cmd))


def cloud_train():
  """Not Implemented Yet"""
  print 'cloud_train'



def local_predict():
  """Not Implemented Yet"""
  print 'local_predict'


def cloud_predict():
  """Not Implemented Yet"""
  print 'cloud_predict'


def local_batch_predict():
  """Not Implemented Yet"""
  print 'local_batch_predict'

def cloud_batch_predict():
  """Not Implemented Yet"""
  print 'cloud_batch_predict'

