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


"""Provides interface for Datalab.

  Datalab will look for functions with the below names:
     local_preprocess
     local_train
     local_predict
     cloud_preprocess
     cloud_train
     cloud_predict
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib
import json

import tensorflow as tf
import yaml

from . import preprocess
from . import trainer

_TF_GS_URL = 'gs://cloud-datalab/deploy/tf/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl'

# TODO(brandondutra): move this url someplace else.
_SD_GS_URL = 'gs://cloud-ml-dev_bdt/structured_data-0.1.tar.gz'


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

def _assert_gcs_files(files):
  """Check files starts wtih gs://.

  Args:
    files: string to file path, or list of file paths.
  """
  if isinstance(files, basestring):
    files = [files]

  for f in files:
    if f is not None and not f.startswith('gs://'):
      raise ValueError('File %s is not a gcs path' % f)


def _run_cmd(cmd):
  output = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

  while True:
    line = output.stdout.readline().rstrip()
    print(line)
    if line == '' and output.poll() != None:
      break


def local_preprocess(output_dir, input_feature_file, input_file_pattern, schema_file):
  """Preprocess data locally with Pandas

  Produce analysis used by training.

  Args:
    output_dir: The output directory to use.
    input_feature_file: Describes defaults and column types.
    input_file_pattern: String. File pattern what will expand into a list of csv
        files.
    schema_file: File path to the schema file.
    
  """
  args = ['local_preprocess',
          '--input_file_pattern=%s' % input_file_pattern,
          '--output_dir=%s' % output_dir,
          '--schema_file=%s' % schema_file,
          '--input_feature_file=%s' % input_feature_file]

  print('Starting local preprocessing.')
  preprocess.local_preprocess.main(args)
  print('Local preprocessing done.')

def cloud_preprocess(output_dir, input_feature_file, input_file_pattern=None, schema_file=None, bigquery_table=None, project_id=None):
  """Preprocess data in the cloud with BigQuery.

  Produce analysis used by training.

  Args:
    output_dir: The output directory to use.
    input_feature_file: Describes defaults and column types.
    input_file_path: String. File pattern what will expand into a list of csv
        files.
    schema_file: File path to the schema file.
    bigquery_table: bigquery name in the form 'dataset.tabele_name'
    project_id: project id the table is in. If none, uses the default project.
  """
  _assert_gcs_files([output_dir, input_file_pattern])

  args = ['cloud_preprocess',
          '--output_dir=%s' % output_dir,
          '--input_feature_file=%s' % input_feature_file]

  if input_file_pattern:
    args.append('--input_file_pattern=%s' % input_file_pattern)
  if schema_file:
    args.append('--schema_file=%s' % schema_file)
  if bigquery_table:
    if not project_id:
      project_id = _default_project()
    full_name = project_id + ':' + bigquery_table
    args.append('--bigquery_table=%s' % full_name)
  
  print('Starting cloud preprocessing.')
  preprocess.cloud_preprocess.main(args)
  print('Cloud preprocessing done.')


def local_train(train_file_pattern,
                eval_file_pattern, 
                preprocess_output_dir, 
                output_dir,
                transforms_file,
                model_type,
                max_steps,
                layer_sizes=None):
  """Train model locally.
  Args:
    train_file_pattern: train csv file
    eval_file_pattern: eval csv file
    preprocess_output_dir:  The output directory from preprocessing
    output_dir:  Output directory of training.
    transforms_file: File path to the transforms file.
    model_type: model type
    max_steps: Int. Number of training steps to perform.
    layer_sizes: List. Represents the layers in the connected DNN.
        If the model type is DNN, this must be set. Example [10, 3, 2], this 
        will create three DNN layers where the first layer will have 10 nodes, 
        the middle layer will have 3 nodes, and the laster layer will have 2
        nodes.
  """
  #TODO(brandondutra): allow other flags to be set like batch size/learner rate
  #TODO(brandondutra): doc someplace that TF>=0.12 and cloudml >-1.7 are needed.

  args = ['local_train',
          '--train_data_paths=%s' % train_file_pattern,
          '--eval_data_paths=%s' % eval_file_pattern,
          '--output_path=%s' % output_dir,
          '--preprocess_output_dir=%s' % preprocess_output_dir,
          '--transforms_file=%s' % transforms_file,
          '--model_type=%s' % model_type,
          '--max_steps=%s' % str(max_steps)]
  if layer_sizes:
    args.extend(['--layer_sizes'] + [str(x) for x in layer_sizes])

  print('Starting local training.')
  trainer.task.main(args)
  print('Local training done.')

def cloud_train(train_file_pattern,
                eval_file_pattern, 
                preprocess_output_dir, 
                output_dir,
                transforms_file,
                model_type,
                max_steps,
                layer_sizes=None,
                staging_bucket=None, 
                project_id=None,
                job_name=None,
                scale_tier='STANDARD_1',
                region=None):
  """Train model using CloudML.
  Args:

    train_file_pattern: train csv file
    eval_file_pattern: eval csv file
    preprocess_output_dir:  The output directory from preprocessing
    output_dir:  Output directory of training.
    transforms_file: File path to the transforms file.
    model_type: model type
    max_steps: Int. Number of training steps to perform.
    layer_sizes: List. Represents the layers in the connected DNN.
        If the model type is DNN, this must be set. Example [10, 3, 2], this 
        will create three DNN layers where the first layer will have 10 nodes, 
        the middle layer will have 3 nodes, and the laster layer will have 2
        nodes.

    staging_bucket: GCS bucket.
    project_id: String. The GCE project to use. Defaults to the notebook's
        default project id.
    job_name: String. Job name as listed on the Dataflow service. If None, a
        default job name is selected.
    scale_tier: The CloudML scale tier. CUSTOM tiers are currently not supported
        in this package. See https://cloud.google.com/ml/reference/rest/v1beta1/projects.jobs#ScaleTier
  """
  #TODO(brandondutra): allow other flags to be set like batch size,
  #   learner rate, custom scale tiers, etc
  #TODO(brandondutra): doc someplace that TF>=0.12 and cloudml >-1.7 are needed.

  _assert_gcs_files([train_file_pattern, eval_file_pattern, 
                     preprocess_output_dir, transforms_file])

  args = ['--train_data_paths=%s' % train_file_pattern,
          '--eval_data_paths=%s' % eval_file_pattern,
          '--output_path=%s' % output_dir,
          '--preprocess_output_dir=%s' % preprocess_output_dir,
          '--transforms_file=%s' % transforms_file,
          '--model_type=%s' % model_type,
          '--max_steps=%s' % str(max_steps)]
  if layer_sizes:
    args.extend(['--layer_sizes'] + [str(x) for x in layer_sizes])

  # TODO(brandondutra): move these package uris locally, ask for a staging
  # and copy them there. This package should work without cloudml having to 
  # maintain gs files!!!
  job_request = {
    'package_uris': [_TF_GS_URL, _SD_GS_URL],
    'python_module': 'datalab_solutions.structured_data.trainer.task',
    'scale_tier': scale_tier,
    'region': region,
    'args': args
  }
  # Local import because cloudml service does not have datalab
  import datalab
  cloud_runner = datalab.mlalpha.CloudRunner(job_request)

  # TODO(brandondutra) update CloudRunner to not mess with the args, so that
  # this hack will not be needed.
  cloud_runner._job_request['args'] = args

  if not job_name:
    job_name = 'structured_data_train_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')
  job_request = cloud_runner.run(job_name)

  
  if _is_in_IPython():
    import IPython
    url = ('https://console.developers.google.com/ml/jobs?project=%s'
                    % _default_project())
    nice_json = json.dumps(job_request, indent=2, separators=(',', ': '))
    html = ('Job Request Sent:<br /><pre>%s</pre>'
            '<p>Click <a href="%s" target="_blank">here</a> to track '
            'the training job %s.</p><br/>' % 
            (nice_json, url, job_name))
    IPython.display.display_html(html, raw=True)
  else:
    print('Job Request Sent:')
    print(job_request)


def local_predict():
  """Runs local prediction.

  Runs local prediction in memory and prints the results to the screen. For
  running prediction on a large dataset or saving the results, run
  local_batch_prediction or batch_prediction.

  Args:

  """
  # Save the instances to a file, call local batch prediction, and print it back




def cloud_predict():
  """Use Online prediction.

  Runs online prediction in the cloud and prints the results to the screen. For
  running prediction on a large dataset or saving the results, run
  local_batch_prediction or batch_prediction.

  Args:


  Before using this, the model must be created. This can be done by running
  two gcloud commands:
  1) gcloud beta ml models create NAME
  2) gcloud beta ml models versions create VERSION --model NAME \
      --origin gs://BUCKET/training_output_dir/model
  or one datalab magic:
  1) %mlalpha deploy --name=NAME.VERSION \
      --path=gs://BUCKET/training_output_dir/model \
      --project=PROJECT
  Note that the model must be on GCS.
  """
  pass



def local_batch_predict(model_dir, prediction_input_file, output_dir,
                        batch_size=1000, shard_files=True):
  """Local batch prediction.

  Args:
    model_dir: local file path to trained model. Usually, this is 
        training_output_dir/model.
    prediction_input_file: csv file pattern to a local file.
    output_dir: local output location to save the results.
    batch_size: Int. How many instances to run in memory at once. Larger values
        mean better performace but more memeory consumed.
    shard_files: If false, the output files are not shardded.
  """
  cmd = ['predict.py',
         '--predict_data=%s' % prediction_input_file,
         '--trained_model_dir=%s' % model_dir,
         '--output_dir=%s' % output_dir,
         '--output_format=csv',
         '--batch_size=%s' % str(batch_size)]

  if shard_files:
    cmd.append('--shard_files')
  else:
    cmd.append('--no-shard_files')

  print('Starting local batch prediction.')
  predict.predict.main(args)
  print('Local batch prediction done.')



def cloud_batch_predict(model_dir, prediction_input_file, output_dir,
                        batch_size=1000, shard_files=True):
  """Cloud batch prediction. Submitts a Dataflow job.

  Args:
    model_dir: GSC file path to trained model. Usually, this is 
        training_output_dir/model.
    prediction_input_file: csv file pattern to a GSC file.
    output_dir: Location to save the results on GCS.
    batch_size: Int. How many instances to run in memory at once. Larger values
        mean better performace but more memeory consumed.
    shard_files: If false, the output files are not shardded.
  """
  cmd = ['predict.py',
         '--cloud',
         '--project_id=%s' % _default_project(),
         '--predict_data=%s' % prediction_input_file,
         '--trained_model_dir=%s' % model_dir,
         '--output_dir=%s' % output_dir,
         '--output_format=csv',
         '--batch_size=%s' % str(batch_size)]

  if shard_files:
    cmd.append('--shard_files')
  else:
    cmd.append('--no-shard_files')

  print('Starting cloud batch prediction.')
  predict.predict.main(args)
  print('See above link for job status.')