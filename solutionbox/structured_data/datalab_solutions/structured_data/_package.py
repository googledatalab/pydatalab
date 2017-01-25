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

import tensorflow as tf
import yaml

import google.cloud.ml as ml

_TF_GS_URL = 'gs://cloud-datalab/deploy/tf/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl'


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


def _run_cmd(cmd):
  output = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

  while True:
    line = output.stdout.readline().rstrip()
    print(line)
    if line == '' and output.poll() != None:
      break


def local_preprocess(input_file_path, output_dir, schema_file,
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
    schema_file: File path to the schema file.
    train_percent: Int in range [0, 100].
    eval_percent: Int in range [0, 100].
    train_percent: Int in range [0, 100].
  """


  percent_flags = _percent_flags(train_percent, eval_percent, test_percent)
  this_folder = os.path.dirname(os.path.abspath(__file__))

  cmd = ['python',
         os.path.join(this_folder, 'preprocess/preprocess.py'),
         '--input_file_path=%s' % input_file_path,
         '--output_dir=%s' % output_dir,
         '--schema_file=%s' % schema_file] + percent_flags

  print('Local preprocess, running command: %s' % ' '.join(cmd))
  _run_cmd(' '.join(cmd))

  print('Local preprocessing done.')


def cloud_preprocess(input_file_path, output_dir, schema_file,
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
    output_dir: The output directory to use; should be GCS path.
    schema_file: File path to the schema file.
    train_percent: Int in range [0, 100].
    eval_percent: Int in range [0, 100].
    train_percent: Int in range [0, 100].
    project_id: String. The GCE project to use. Defaults to the notebook's
        default project id.
    job_name: String. Job name as listed on the Dataflow service. If None, a
        default job name is selected.
  """

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
         '--schema_file=%s' % schema_file] + percent_flags

  print('Cloud preprocess, running command: %s' % ' '.join(cmd))
  _run_cmd(' '.join(cmd))

  print('Cloud preprocessing job submitted.')

  if _is_in_IPython():
    import IPython

    dataflow_url = ('https://console.developers.google.com/dataflow?project=%s'
                    % project_id)
    html = ('<p>Click <a href="%s" target="_blank">here</a> to track '
            'preprocessing job %s.</p><br/>' % (dataflow_url, job_name))
    IPython.display.display_html(html, raw=True)



def local_train(preprocessed_dir, schema_file, transforms_file, output_dir,
                model_type,
                layer_sizes=None, max_steps=None):
  """Train model locally.
  Args:
    preprocessed_dir: The output directory from preprocessing. Must contain
        files named features_train*.tfrecord.gz, features_eval*.tfrecord.gz,
        and metadata.json. Can be local or GCS path.
    schema_file: Same file used for preprocessing
    transforms_file: File describing transforms to perform on the data.
    output_dir: Output directory of training.
    layer_sizes: String. Represents the layers in the connected DNN.
        If the model type is DNN, this must be set. Example "10 3 2", this will
        create three DNN layers where the first layer will have 10 nodes, the
        middle layer will have 3 nodes, and the laster layer will have 2 nodes.
    max_steps: Int. Number of training steps to perform.
  """
  #_check_transforms_config_file(transforms_config_file)

  #TODO(brandondutra): allow other flags to be set like batch size/learner rate
  #TODO(brandondutra): doc someplace that TF>=0.12 and cloudml >-1.7 are needed.

  train_filename = os.path.join(preprocessed_dir, 'features_train*')
  eval_filename = os.path.join(preprocessed_dir, 'features_eval*')
  metadata_filename = os.path.join(preprocessed_dir, 'metadata.json')
  this_folder = os.path.dirname(os.path.abspath(__file__))

  #TODO(brandondutra): remove the cd after b/34221856
  cmd = ['cd %s &&' % this_folder,
         'gcloud beta ml local train',
         '--module-name=trainer.task',
         '--package-path=trainer',
         '--',
         '--train_data_paths=%s' % train_filename,
         '--eval_data_paths=%s' % eval_filename,
         '--metadata_path=%s' % metadata_filename,
         '--output_path=%s' % output_dir,
         '--schema_file=%s' % schema_file,
         '--transforms_file=%s' % transforms_file,
         '--model_type=%s' % model_type,
         '--max_steps=%s' % str(max_steps)]
  if layer_sizes:
    cmd += ['--layer_sizes %s' % layer_sizes]

  print('Local training, running command: %s' % ' '.join(cmd))
  _run_cmd(' '.join(cmd))
  print('Local training done.')


def cloud_train(preprocessed_dir, schema_file, transforms_file, output_dir,
                model_type, staging_bucket,
                layer_sizes=None, max_steps=None, project_id=None,
                job_name=None, scale_tier='BASIC'):
  """Train model using CloudML.
  Args:
    preprocessed_dir: The output directory from preprocessing. Must contain
        files named features_train*.tfrecord.gz, features_eval*.tfrecord.gz,
        and metadata.json.
    schema_file: File path to the schema file.
    transforms_file: File path to the transforms file.
    output_dir: Output directory of training.
    staging_bucket: GCS bucket.
    layer_sizes: String. Represents the layers in the connected DNN.
        If the model type is DNN, this must be set. Example "10 3 2", this will
        create three DNN layers where the first layer will have 10 nodes, the
        middle layer will have 3 nodes, and the laster layer will have 2 nodes.
    max_steps: Int. Number of training steps to perform.
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

  if (not preprocessed_dir.startswith('gs://')
      or not transforms_file.startswith('gs://')
      or not schema_file.startswith('gs://')
      or not output_dir.startswith('gs://')):
    print('ERROR: preprocessed_dir, transforms_file, output_dir, '
          'and schema_file must all be in GCS.')
    return

  # Training will fail if there are files in the output folder. Check now and
  # fail fast.
  if ml.util._file.glob_files(os.path.join(output_dir, '*')):
    print('ERROR: output_dir should be empty. Use another folder')
    return

  #TODO(brandondutra): remove the tf stuff once the cloudml service is past 0.11
  temp_dir = tempfile.mkdtemp()
  subprocess.check_call(['gsutil', 'cp', _TF_GS_URL, temp_dir])
  tf_local_package = os.path.join(temp_dir, os.path.basename(_TF_GS_URL))

  # Bulid the training config file.
  training_config_file_path = tempfile.mkstemp(dir=temp_dir)[1]
  training_config = {'trainingInput': {'scaleTier': scale_tier}}
  with open(training_config_file_path, 'w') as f:
    f.write(yaml.dump(training_config, default_flow_style=False))

  train_filename = os.path.join(preprocessed_dir, 'features_train*')
  eval_filename = os.path.join(preprocessed_dir, 'features_eval*')
  metadata_filename = os.path.join(preprocessed_dir, 'metadata.json')
  this_folder = os.path.dirname(os.path.abspath(__file__))
  project_id = project_id or _default_project()
  job_name = job_name or ('structured_data_train_' +
                          datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

  # TODO(brandondutra): remove the cd after b/34221856
  cmd = ['cd %s &&' % this_folder,
         'gcloud beta ml jobs submit training %s' % job_name,
         '--module-name=trainer.task',
         '--staging-bucket=%s' % staging_bucket,
         '--async',
         '--package-path=%s' % 'trainer', #os.path.join(this_folder, 'trainer'),
         '--packages=%s' % tf_local_package,
         '--config=%s' % training_config_file_path,
         '--',
         '--train_data_paths=%s' % train_filename,
         '--eval_data_paths=%s' % eval_filename,
         '--metadata_path=%s' % metadata_filename,
         '--output_path=%s' % output_dir,
         '--transforms_file=%s' % transforms_file,
         '--schema_file=%s' % schema_file,
         '--model_type=%s' % model_type,
         '--max_steps=%s' % str(max_steps)]
  if layer_sizes:
    cmd += ['--layer_sizes %s' % layer_sizes]

  print('CloudML training, running command: %s' % ' '.join(cmd))
  _run_cmd(' '.join(cmd))

  print('CloudML training job submitted.')

  if _is_in_IPython():
    import IPython

    dataflow_url = ('https://console.developers.google.com/ml/jobs?project=%s'
                    % project_id)
    html = ('<p>Click <a href="%s" target="_blank">here</a> to track '
            'the training job %s.</p><br/>' % (dataflow_url, job_name))
    IPython.display.display_html(html, raw=True)

  # Delete the temp files made
  shutil.rmtree(temp_dir)

def local_predict(model_dir, prediction_input_file):
  """Runs local prediction.

  Runs local prediction in memory and prints the results to the screen. For
  running prediction on a large dataset or saving the results, run
  local_batch_prediction or batch_prediction.

  Args:
    model_dir: Path to folder that contains the model. This is usully OUT/model
        where OUT is the value of output_dir when local_training was ran.
    prediction_input_file: csv file that has the same schem as the input
        files used during local_preprocess, except that the target column is
        removed.
  """
  #TODO(brandondutra): remove this hack once cloudml 1.8 is released.
  # Check that the model folder has a metadata.yaml file. If not, copy it.
  if not os.path.isfile(os.path.join(model_dir, 'metadata.yaml')):
    shutil.copy2(os.path.join(model_dir, 'metadata.json'),
                 os.path.join(model_dir, 'metadata.yaml'))


  cmd = ['gcloud beta ml local predict',
         '--model-dir=%s' % model_dir,
         '--text-instances=%s' % prediction_input_file]

  _run_cmd(' '.join(cmd))
  print('Local prediction done.')


def cloud_predict(model_name, prediction_input_file, version_name=None):
  """Use Online prediction.

  Runs online prediction in the cloud and prints the results to the screen. For
  running prediction on a large dataset or saving the results, run
  local_batch_prediction or batch_prediction.

  Args:
    model_dir: Path to folder that contains the model. This is usully OUT/model
        where OUT is the value of output_dir when local_training was ran.
    prediction_input_file: csv file that has the same schem as the input
        files used during local_preprocess, except that the target column is
        removed.
    vsersion_name: Optional version of the model to use. If None, the default
        version is used.

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
  cmd = ['gcloud beta ml predict',
         '--model=%s' % model_name,
         '--text-instances=%s' % prediction_input_file]
  if version_name:
    cmd += ['--version=%s' % version_name]

  print('CloudML online prediction, running command: %s' % ' '.join(cmd))
  _run_cmd(' '.join(cmd))
  print('CloudML online prediction done.')


def local_batch_predict(model_dir, prediction_input_file, output_dir):
  """Local batch prediction.

  Args:
    model_dir: local path to trained model.
    prediction_input_file: File path to input files. May contain a file pattern.
        Only csv files are supported, and the scema must match what was used
        in preprocessing except that the target column is removed.
    output_dir: folder to save results to.
  """

  #TODO(brandondutra): remove this hack once cloudml 1.8 is released.
  # Check that the model folder has a metadata.yaml file. If not, copy it.
  if not os.path.isfile(os.path.join(model_dir, 'metadata.yaml')):
    shutil.copy2(os.path.join(model_dir, 'metadata.json'),
                 os.path.join(model_dir, 'metadata.yaml'))


  cmd = ['python -m google.cloud.ml.dataflow.batch_prediction_main',
         '--input_file_format=text',
         '--input_file_patterns=%s' % prediction_input_file,
         '--output_location=%s' % output_dir,
         '--model_dir=%s' % model_dir]

  print('Local batch prediction, running command: %s' % ' '.join(cmd))
  _run_cmd(' '.join(cmd))
  print('Local batch prediction done.')


def cloud_batch_predict(model_name, prediction_input_file, output_dir, region,
                        job_name=None, version_name=None):
  """Cloud batch prediction.

  Args:
    model_name: name of the model. The model must already exist.
    prediction_input_file: File path to input files. May contain a file pattern.
        Only csv files are supported, and the scema must match what was used
        in preprocessing except that the target column is removed. Files must
        be on GCS
    output_dir: GCS folder to safe results to.
    region: GCP compute region to run the batch job. Try using your default
        region first, as this cloud batch prediction is not avaliable in all
        regions.
    job_name: job name used for the cloud job.
    version_name: model version to use. If node, the default version of the
        model is used.
    """

  job_name = job_name or ('structured_data_batch_predict_' +
                          datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

  if (not prediction_input_file.startswith('gs://') or
      not output_dir.startswith('gs://')):
    print('ERROR: prediction_input_file and output_dir must point to a '
          'location on GCS.')
    return

  cmd = ['gcloud beta ml jobs submit prediction %s' % job_name,
         '--model=%s' % model_name,
         '--region=%s' % region,
         '--data-format=TEXT',
         '--input-paths=%s' % prediction_input_file,
         '--output-path=%s' % output_dir]
  if version_name:
    cmd += ['--version=%s' % version_name]

  print('CloudML batch prediction, running command: %s' % ' '.join(cmd))
  _run_cmd(' '.join(cmd))
  print('CloudML batch prediction job submitted.')

  if _is_in_IPython():
    import IPython

    dataflow_url = ('https://console.developers.google.com/ml/jobs?project=%s'
                    % _default_project())
    html = ('<p>Click <a href="%s" target="_blank">here</a> to track '
            'the prediction job %s.</p><br/>' % (dataflow_url, job_name))
    IPython.display.display_html(html, raw=True)


