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
import glob
import StringIO
import subprocess
import uuid
import pandas as pd
import tensorflow as tf
from tensorflow.python.lib.io import file_io

# Note that subpackages of _structured_data are locally imported.
# This is because this part of the mltoolbox is packaged up during training
# and batch prediction. This package depends on datafow, but the trainer
# workers should not need it; therefore at training time, a package cannot
# import apache_beam.

# Likewise, datalab packages are locally imported. This is because TF and
# dataflow workers should not need it.


_TF_GS_URL = 'gs://cloud-datalab/deploy/tf/tensorflow-1.0.0-cp27-cp27mu-manylinux1_x86_64.whl'
_PROTOBUF_GS_URL = 'gs://cloud-datalab/deploy/tf/protobuf-3.1.0-py2.py3-none-any.whl'


def _default_project():
  from google.datalab import Context
  return Context.default().project_id

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


def _package_to_staging(staging_package_url):
    """Repackage this package from local installed location and copy it to GCS.

    Args:
      staging_package_url: GCS path.
    """
    import google.datalab.ml as ml

    # Find the package root. __file__ is under [package_root]/mltoolbox/_structured_data/this_file
    package_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../'))
    setup_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'master_setup.py'))
    tar_gz_path = os.path.join(staging_package_url, 'staging', 'trainer.tar.gz')

    print('Building package and uploading to %s' % tar_gz_path)
    ml.package_and_copy(package_root, setup_path, tar_gz_path)

    return tar_gz_path


def _wait_and_kill(pid_to_wait, pids_to_kill):
  """ Helper function.

      Wait for a process to finish if it exists, and then try to kill a list of
      processes.

      Used by local_train

  Args:
    pid_to_wait: the process to wait for.
    pids_to_kill: a list of processes to kill after the process of pid_to_wait finishes.
  """
  # cloud workers don't have psutil
  import psutil
  if psutil.pid_exists(pid_to_wait):
    psutil.Process(pid=pid_to_wait).wait()

  for pid_to_kill in pids_to_kill:
    if psutil.pid_exists(pid_to_kill):
      p = psutil.Process(pid=pid_to_kill)
      p.kill()
      p.wait()


# ==============================================================================
# Analyze
# ==============================================================================

def analyze(output_dir, dataset, cloud=False, project_id=None):
  """Blocking version of analyze_async. See documentation of analyze_async."""
  job = analyze_async(
      output_dir=output_dir,
      dataset=dataset,
      cloud=cloud,
      project_id=project_id)
  job.wait()
  print('Analyze: ' + str(job.state))


def analyze_async(output_dir, dataset, cloud=False, project_id=None):
  """Analyze data locally or in the cloud with BigQuery.

  Produce analysis used by training. This can take a while, even for small
  datasets. For small datasets, it may be faster to use local_analysis.

  Args:
    output_dir: The output directory to use.
    dataset: only CsvDataSet is supported currently.
    cloud: If False, runs analysis locally with Pandas. If Ture, runs analysis
        in the cloud with BigQuery.
    project_id: Uses BigQuery with this project id. Default is datalab's 
        default project id.

  Returns:
    A google.datalab.utils.Job object that can be used to query state from or wait.
  """
  import google.datalab.utils as du
  fn = lambda : _analyze(output_dir, dataset, cloud, project_id)
  return du.LambdaJob(fn, job_id=None)  

def _analyze(output_dir, dataset, cloud=False, project_id=None):
  import google.datalab.ml as ml
  from . import preprocess 

  if not isinstance(dataset, ml.CsvDataSet):
    raise ValueError('Only CsvDataSet is supported')

  if len(dataset.input_files) != 1:
    raise ValueError('CsvDataSet should be built with a file pattern, not a '
                     'list of files.')

  if project_id and not cloud:
    raise ValueError('project_id only needed if cloud is True')

  if cloud:
    _assert_gcs_files([output_dir, dataset.input_files[0]])


  tmp_dir = tempfile.mkdtemp()
  try:
    # write the schema file.
    _, schema_file_path = tempfile.mkstemp(dir=tmp_dir, suffix='.json',
                                           prefix='schema')    
    file_io.write_string_to_file(schema_file_path, json.dumps(dataset.schema))

    args = ['preprocess',
            '--input-file-pattern=%s' % dataset.input_files[0],
            '--output-dir=%s' % output_dir,
            '--schema-file=%s' % schema_file_path]

    if cloud:
      print('Track BigQuery status at')
      print('https://bigquery.cloud.google.com/queries/%s' % _default_project())
      preprocess.cloud_preprocess.main(args)
    else:
      preprocess.local_preprocess.main(args)
  finally:
    shutil.rmtree(tmp_dir)


# ==============================================================================
# Train
# ==============================================================================
def train_async(train_dataset,
          eval_dataset,
          analysis_dir,
          output_dir,
          features,
          model_type,
          max_steps=5000,
          num_epochs=None,
          train_batch_size=100,
          eval_batch_size=16,
          min_eval_frequency=100,
          top_n=None,
          layer_sizes=None,
          learning_rate=0.01,
          epsilon=0.0005,
          job_name=None, # cloud param
          job_name_prefix='', # cloud param
          cloud=None, # cloud param
          ):
  # NOTE: if you make a chane go this doc string, you MUST COPY it 4 TIMES in 
  # mltoolbox.{classification|regression}.{dnn|linear}, but you must remove
  # the model_type parameter, and maybe change the layer_sizes and top_n
  # parameters!
  # Datalab does some tricky things and messing with train.__doc__ will
  # not work!
  """Train model locally or in the cloud.

  Local Training:

  Args:
    train_dataset: CsvDataSet
    eval_dataset: CsvDataSet
    analysis_dir:  The output directory from local_analysis
    output_dir:  Output directory of training.
    features: file path or features object. Example:
        {
          "col_A": {"transform": "scale", "default": 0.0},
          "col_B": {"transform": "scale","value": 4},
          # Note col_C is missing, so default transform used.
          "col_D": {"transform": "hash_one_hot", "hash_bucket_size": 4},
          "col_target": {"transform": "target"},
          "col_key": {"transform": "key"}
        }
        The keys correspond to the columns in the input files as defined by the
        schema file during preprocessing. Some notes
        1) The "key" and "target" transforms are required.
        2) Default values are optional. These are used if the input data has
           missing values during training and prediction. If not supplied for a
           column, the default value for a numerical column is that column's
           mean vlaue, and for a categorical column the empty string is used.
        3) For numerical colums, the following transforms are supported:
           i) {"transform": "identity"}: does nothing to the number. (default)
           ii) {"transform": "scale"}: scales the colum values to -1, 1.
           iii) {"transform": "scale", "value": a}: scales the colum values
              to -a, a.

           For categorical colums, the following transforms are supported:
          i) {"transform": "one_hot"}: A one-hot vector using the full
              vocabulary is used. (default)
          ii) {"transform": "embedding", "embedding_dim": d}: Each label is
              embedded into an d-dimensional space.
    model_type: One of 'linear_classification', 'linear_regression',
        'dnn_classification', 'dnn_regression'.
    max_steps: Int. Number of training steps to perform.
    num_epochs: Maximum number of training data epochs on which to train.
        The training job will run for max_steps or num_epochs, whichever occurs
        first.
    train_batch_size: number of rows to train on in one step.
    eval_batch_size: number of rows to eval in one step. One pass of the eval
        dataset is done. If eval_batch_size does not perfectly divide the numer
        of eval instances, the last fractional batch is not used.
    min_eval_frequency: Minimum number of training steps between evaluations.
    top_n: Int. For classification problems, the output graph will contain the
        labels and scores for the top n classes with a default of n=1. Use
        None for regression problems.
    layer_sizes: List. Represents the layers in the connected DNN.
        If the model type is DNN, this must be set. Example [10, 3, 2], this
        will create three DNN layers where the first layer will have 10 nodes,
        the middle layer will have 3 nodes, and the laster layer will have 2
        nodes.
    learning_rate: tf.train.AdamOptimizer's learning rate,
    epsilon: tf.train.AdamOptimizer's epsilon value.

  Cloud Training:

  Args:
    All local training arguments are valid for cloud training. Cloud training
    contains two additional args:

    cloud: A CloudTrainingConfig object.
    job_name: Training job name. A default will be picked if None. 
    job_name_prefix: If job_name is None, the job will be named 
        '<job_name_prefix>_<timestamp>'.   

  Returns:
    A google.datalab.utils.Job object that can be used to query state from or wait.
  """
  import google.datalab.utils as du
  
  if model_type not in ['linear_classification', 'linear_regression',
      'dnn_classification', 'dnn_regression']:
    raise ValueError('Unknown model_type %s' % model_type)

  if cloud:
    return cloud_train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        features=features,
        model_type=model_type,
        max_steps=max_steps,
        num_epochs=num_epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        min_eval_frequency=min_eval_frequency,
        top_n=top_n,
        layer_sizes=layer_sizes,
        learning_rate=learning_rate,
        epsilon=epsilon,
        job_name=job_name,
        job_name_prefix=job_name_prefix,
        config=cloud,      
    )
  else:
    def fn():
      return local_train(
          train_dataset=train_dataset,
          eval_dataset=eval_dataset,
          analysis_dir=analysis_dir,
          output_dir=output_dir,
          features=features,
          model_type=model_type,
          max_steps=max_steps,
          num_epochs=num_epochs,
          train_batch_size=train_batch_size,
          eval_batch_size=eval_batch_size,
          min_eval_frequency=min_eval_frequency,
          top_n=top_n,
          layer_sizes=layer_sizes,
          learning_rate=learning_rate,
          epsilon=epsilon)
    return du.LambdaJob(fn, job_id=None)  

def local_train(train_dataset,
                eval_dataset,
                analysis_dir,
                output_dir,
                features,
                model_type,
                max_steps,
                num_epochs,
                train_batch_size,
                eval_batch_size,
                min_eval_frequency,
                top_n,
                layer_sizes,
                learning_rate,
                epsilon):
  if len(train_dataset.input_files) != 1 or len(eval_dataset.input_files) != 1:
    raise ValueError('CsvDataSets must be built with a file pattern, not list '
                     'of files.')

  if file_io.file_exists(output_dir):
    raise ValueError('output_dir already exist. Use a new output path.')

  if eval_dataset.size < eval_batch_size:
    raise ValueError('Eval batch size must be smaller than the eval data size.')

  if isinstance(features, dict):
    # Make a features file.
    if not file_io.file_exists(output_dir):
      file_io.recursive_create_dir(output_dir)
    features_file = os.path.join(output_dir, 'features_file.json')
    file_io.write_string_to_file(
        features_file,
        json.dumps(features))
  else:
    features_file = features

  def _get_abs_path(input_path):
    cur_path = os.getcwd()
    full_path = os.path.abspath(os.path.join(cur_path, input_path))
    # put path in quotes as it could contain spaces.
    return "'" + full_path + "'"

  args = ['cd %s &&' % os.path.abspath(os.path.dirname(__file__)),
          'python -m trainer.task',
          '--train-data-paths=%s' % _get_abs_path(train_dataset.input_files[0]),
          '--eval-data-paths=%s' % _get_abs_path(eval_dataset.input_files[0]),
          '--job-dir=%s' % _get_abs_path(output_dir),
          '--preprocess-output-dir=%s' % _get_abs_path(analysis_dir),
          '--transforms-file=%s' % _get_abs_path(features_file),
          '--model-type=%s' % model_type,
          '--max-steps=%s' % str(max_steps),
          '--train-batch-size=%s' % str(train_batch_size),
          '--eval-batch-size=%s' % str(eval_batch_size),
          '--min-eval-frequency=%s' % str(min_eval_frequency),
          '--learning-rate=%s' % str(learning_rate),
          '--epsilon=%s' % str(epsilon)]
  if num_epochs:
    args.append('--num-epochs=%s' % str(num_epochs))
  if top_n:
    args.append('--top-n=%s' % str(top_n))
  if layer_sizes:
    for i in range(len(layer_sizes)):
      args.append('--layer-size%s=%s' % (i+1, str(layer_sizes[i])))

  monitor_process = None
  try:
    p = subprocess.Popen(' '.join(args),
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    pids_to_kill = [p.pid]

    #script -> name = datalab_structured_data._package
    script = 'import %s; %s._wait_and_kill(%s, %s)' % \
          (__name__, __name__, str(os.getpid()), str(pids_to_kill))
    monitor_process = subprocess.Popen(['python', '-c', script])

    while p.poll() is None:
      line = p.stdout.readline()
      if (line.startswith('INFO:tensorflow:global') or
          line.startswith('INFO:tensorflow:loss') or
          line.startswith('INFO:tensorflow:Saving dict')):
        sys.stdout.write(line)
  finally:
    if monitor_process:
      monitor_process.kill()
      monitor_process.wait()
  

def cloud_train(train_dataset,
                eval_dataset,
                analysis_dir,
                output_dir,
                features,
                model_type,
                max_steps,
                num_epochs,
                train_batch_size,
                eval_batch_size,
                min_eval_frequency,
                top_n,
                layer_sizes,
                learning_rate,
                epsilon,
                job_name,
                job_name_prefix,
                config):
  """Train model using CloudML.

  See local_train() for a description of the args.
  Args:
    config: A CloudTrainingConfig object.
    job_name: Training job name. A default will be picked if None.
  """
  import google.datalab.ml as ml

  if len(train_dataset.input_files) != 1 or len(eval_dataset.input_files) != 1:
    raise ValueError('CsvDataSets must be built with a file pattern, not list '
                     'of files.')

  if file_io.file_exists(output_dir):
    raise ValueError('output_dir already exist. Use a new output path.')

  if isinstance(features, dict):
    # Make a features file.
    if not file_io.file_exists(output_dir):
      file_io.recursive_create_dir(output_dir)
    features_file = os.path.join(output_dir, 'features_file.json')
    file_io.write_string_to_file(
        features_file,
        json.dumps(features))
  else:
    features_file = features

  if not isinstance(config, ml.CloudTrainingConfig):
    raise ValueError('cloud should be an instance of '
                     'google.datalab.ml.CloudTrainingConfig for cloud training.')

  _assert_gcs_files([output_dir, train_dataset.input_files[0],
      eval_dataset.input_files[0], features_file,
      analysis_dir])

  args = ['--train-data-paths=%s' % train_dataset.input_files[0],
          '--eval-data-paths=%s' % eval_dataset.input_files[0],
          '--preprocess-output-dir=%s' % analysis_dir,
          '--transforms-file=%s' % features_file,
          '--model-type=%s' % model_type,
          '--max-steps=%s' % str(max_steps),
          '--train-batch-size=%s' % str(train_batch_size),
          '--eval-batch-size=%s' % str(eval_batch_size),
          '--min-eval-frequency=%s' % str(min_eval_frequency),
          '--learning-rate=%s' % str(learning_rate),
          '--epsilon=%s' % str(epsilon)]
  if num_epochs:
    args.append('--num-epochs=%s' % str(num_epochs))
  if top_n:
    args.append('--top-n=%s' % str(top_n))
  if layer_sizes:
    for i in range(len(layer_sizes)):
      args.append('--layer-size%s=%s' % (i+1, str(layer_sizes[i])))

  job_request = {
    'package_uris': [_package_to_staging(output_dir), _TF_GS_URL, _PROTOBUF_GS_URL],
    'python_module': 'mltoolbox._structured_data.trainer.task',
    'job_dir': output_dir,
    'args': args
  }
  job_request.update(dict(config._asdict()))

  if not job_name:
    job_name = job_name_prefix or 'structured_data_train'
    job_name += '_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')
  job = ml.Job.submit_training(job_request, job_name)
  print('Job request send. View status of job at')
  print('https://console.developers.google.com/ml/jobs?project=%s' %
        _default_project())

  return job

# ==============================================================================
# Predict
# ==============================================================================

def predict(data, training_dir=None, model_name=None, model_version=None, 
  cloud=False):
  """Runs prediction locally or on the cloud.

  Args:
    data: List of csv strings or a Pandas DataFrame that match the model schema.
    training_dir: local path to the trained output folder.
    model_name: deployed model name
    model_version: depoyed model version
    cloud: bool. If False, does local prediction and data and training_dir
        must be set. If True, does cloud prediction and data, model_name, 
        and model_version must be set.


  For cloud prediction, the model must be created. This can be done by running
  two gcloud commands::
    1) gcloud beta ml models create NAME
    2) gcloud beta ml versions create VERSION --model NAME --origin gs://BUCKET/training_dir/model
  or these datalab commands:
    1) import google.datalab as datalab
      model = datalab.ml.ModelVersions(MODEL_NAME)
      model.deploy(version_name=VERSION, path='gs://BUCKET/training_dir/model')
  Note that the model must be on GCS.

  Returns:
    Pandas DataFrame.
  """
  if cloud:
    if not model_version or not model_name:
      raise ValueError('model_version or model_name is not set')
    if training_dir:
      raise ValueError('training_dir not needed when cloud is True')

    return cloud_predict(model_name, model_version, data)
  else:
    if not training_dir:
      raise ValueError('training_dir is not set')
    if model_version or model_name:
      raise ValueError('model_name and model_version not needed when cloud is '
                       'False.')
    return local_predict(training_dir, data)


def local_predict(training_dir, data):
  """Runs local prediction on the prediction graph.

  Runs local prediction and returns the result in a Pandas DataFrame. For
  running prediction on a large dataset or saving the results, run
  local_batch_prediction or batch_prediction. Input data should fully match
  the schema that was used at training, except the target column should not
  exist.

  Args:
    training_dir: local path to the trained output folder.
    data: List of csv strings or a Pandas DataFrame that match the model schema.

  """
  #from . import predict as predict_module
  from .prediction import predict as predict_module

  # Save the instances to a file, call local batch prediction, and return it
  tmp_dir = tempfile.mkdtemp()
  _, input_file_path = tempfile.mkstemp(dir=tmp_dir, suffix='.csv',
                                        prefix='input')

  try:
    if isinstance(data, pd.DataFrame):
      data.to_csv(input_file_path, header=False, index=False)
    else:
      with open(input_file_path, 'w') as f:
        for line in data:
          f.write(line + '\n')

    model_dir = os.path.join(training_dir, 'model')
    if not file_io.file_exists(model_dir):
      raise ValueError('training_dir should contain the folder model')

    cmd = ['predict.py',
           '--predict-data=%s' % input_file_path,
           '--trained-model-dir=%s' % model_dir,
           '--output-dir=%s' % tmp_dir,
           '--output-format=csv',
           '--batch-size=16',
           '--mode=prediction',
           '--no-shard-files']

    #runner_results = predict_module.predict.main(cmd)
    runner_results = predict_module.main(cmd)
    runner_results.wait_until_finish()

    # Read the header file.
    schema_file = os.path.join(tmp_dir, 'csv_schema.json')
    with open(schema_file, 'r') as f:
      schema = json.loads(f.read())

    # Print any errors to the screen.
    errors_file = glob.glob(os.path.join(tmp_dir, 'errors*'))
    if errors_file and os.path.getsize(errors_file[0]) > 0:
      print('Warning: there are errors. See below:')
      with open(errors_file[0], 'r') as f:
        text = f.read()
        print(text)

    # Read the predictions data.
    prediction_file = glob.glob(os.path.join(tmp_dir, 'predictions*'))
    if not prediction_file:
      raise FileNotFoundError('Prediction results not found')
    predictions = pd.read_csv(prediction_file[0],
                              header=None,
                              names=[col['name'] for col in schema])
    return predictions
  finally:
    shutil.rmtree(tmp_dir)


def cloud_predict(model_name, model_version, data):
  """Use Online prediction.

  Runs online prediction in the cloud and prints the results to the screen. For
  running prediction on a large dataset or saving the results, run
  local_batch_prediction or batch_prediction.

  Args:
    model_name: deployed model name
    model_version: depoyed model version
    data: List of csv strings or a Pandas DataFrame that match the model schema.

  Before using this, the model must be created. This can be done by running
  two gcloud commands:
  1) gcloud beta ml models create NAME
  2) gcloud beta ml versions create VERSION --model NAME \
      --origin gs://BUCKET/training_dir/model
  or these datalab commands:
  1) import google.datalab as datalab
     model = datalab.ml.ModelVersions(MODEL_NAME)
     model.deploy(version_name=VERSION,
                  path='gs://BUCKET/training_dir/model')
  Note that the model must be on GCS.
  """
  import google.datalab.ml as ml

  if isinstance(data, pd.DataFrame):
    # write the df to csv.
    string_buffer = StringIO.StringIO()
    data.to_csv(string_buffer, header=None, index=False)
    input_data = string_buffer.getvalue().split('\n')

    #remove empty strings
    input_data = [line for line in input_data if line]
  else:
    input_data = data

  predictions = ml.ModelVersions(model_name).predict(model_version, input_data)

  # Convert predictions into a dataframe
  df = pd.DataFrame(columns=sorted(predictions[0].keys()))
  for i in range(len(predictions)):
    for k, v in predictions[i].iteritems():
      df.loc[i, k] = v
  return df

# ==============================================================================
# Batch predict
# ==============================================================================

def batch_predict(training_dir, prediction_input_file, output_dir,
                  mode, batch_size=16, shard_files=True, output_format='csv',
                  cloud=False):
  """Blocking versoin of batch_predict. 

  See documentation of batch_prediction_async.
  """
  job = batch_predict_async(
      training_dir=training_dir,
      prediction_input_file=prediction_input_file, 
      output_dir=output_dir,
      mode=mode, 
      batch_size=batch_size, 
      shard_files=shard_files, 
      output_format=output_format,
      cloud=cloud)
  job.wait()
  print('Batch predict: ' + str(job.state))


def batch_predict_async(training_dir, prediction_input_file, output_dir,
                  mode, batch_size=16, shard_files=True, output_format='csv',
                  cloud=False):
  """Local and cloud batch prediction.

  Args:
    training_dir: The output folder of training.
    prediction_input_file: csv file pattern to a file. File must be on GCS if 
        running cloud prediction
    output_dir: output location to save the results. Must be a GSC path if 
        running cloud prediction.
    mode: 'evaluation' or 'prediction'. If 'evaluation', the input data must
        contain a target column. If 'prediction', the input data must not
        contain a target column.
    batch_size: Int. How many instances to run in memory at once. Larger values
        mean better performace but more memeory consumed.
    shard_files: If False, the output files are not shardded.
    output_format: csv or json. Json file are json-newlined.
    cloud: If ture, does cloud batch prediction. If False, runs batch prediction
        locally.

  Returns:
    A google.datalab.utils.Job object that can be used to query state from or wait.
  """
  import google.datalab.utils as du
  if cloud:
    runner_results = cloud_batch_predict(training_dir,
        prediction_input_file, output_dir, mode, batch_size, shard_files,
        output_format)
    job = du.DataflowJob(runner_results)
  else:
    runner_results = local_batch_predict(training_dir,
        prediction_input_file, output_dir, mode, batch_size, shard_files, 
        output_format)
    job = du.LambdaJob(lambda: runner_results.wait_until_finish(),
        job_id=None)

  return job


def local_batch_predict(training_dir, prediction_input_file, output_dir,
                        mode,
                        batch_size, shard_files, output_format):
  """See batch_predict"""
  #from . import predict as predict_module
  from .prediction import predict as predict_module

  if mode == 'evaluation':
    model_dir = os.path.join(training_dir, 'evaluation_model')
  elif mode == 'prediction':
    model_dir = os.path.join(training_dir, 'model')
  else:
    raise ValueError('mode must be evaluation or prediction')

  if not file_io.file_exists(model_dir):
    raise ValueError('Model folder %s does not exist' % model_dir)

  cmd = ['predict.py',
         '--predict-data=%s' % prediction_input_file,
         '--trained-model-dir=%s' % model_dir,
         '--output-dir=%s' % output_dir,
         '--output-format=%s' % output_format,
         '--batch-size=%s' % str(batch_size),
         '--shard-files' if shard_files else '--no-shard-files',
         '--has-target' if mode == 'evaluation' else '--no-has-target'
         ]

  #return predict_module.predict.main(cmd)
  return predict_module.main(cmd)



def cloud_batch_predict(training_dir, prediction_input_file, output_dir,
                        mode,
                        batch_size, shard_files, output_format):
  """See batch_predict"""
  #from . import predict as predict_module
  from .prediction import predict as predict_module

  if mode == 'evaluation':
    model_dir = os.path.join(training_dir, 'evaluation_model')
  elif mode == 'prediction':
    model_dir = os.path.join(training_dir, 'model')
  else:
    raise ValueError('mode must be evaluation or prediction')

  if not file_io.file_exists(model_dir):
    raise ValueError('Model folder %s does not exist' % model_dir)

  _assert_gcs_files([training_dir, prediction_input_file,
      output_dir])

  cmd = ['predict.py',
         '--cloud',
         '--project-id=%s' % _default_project(),
         '--predict-data=%s' % prediction_input_file,
         '--trained-model-dir=%s' % model_dir,
         '--output-dir=%s' % output_dir,
         '--output-format=%s' % output_format,
         '--batch-size=%s' % str(batch_size),
         '--shard-files' if shard_files else '--no-shard-files',
         '--extra-package=%s' % _TF_GS_URL,
         '--extra-package=%s' % _PROTOBUF_GS_URL,
         '--extra-package=%s' % _package_to_staging(output_dir)
         ]

  return predict_module.main(cmd)

