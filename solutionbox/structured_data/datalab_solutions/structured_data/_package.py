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

import pandas as pd
import tensorflow as tf

from tensorflow.python.lib.io import file_io

from . import preprocess
from . import trainer
from . import predict


_TF_GS_URL = 'gs://cloud-datalab/deploy/tf/tensorflow-0.12.0rc1-cp27-none-linux_x86_64.whl'


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


def _package_to_staging(staging_package_url):
    """Repackage this package from local installed location and copy it to GCS.

    Args:
      staging_package_url: GCS path.
    """
    import datalab.mlalpha as mlalpha

    # Find the package root. __file__ is under [package_root]/datalab_solutions/inception.
    package_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../'))
    setup_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'master_setup.py'))
    tar_gz_path = os.path.join(staging_package_url, 'staging', 'sd.tar.gz')

    print('Building package and uploading to %s' % tar_gz_path)
    mlalpha.package_and_copy(package_root, setup_path, tar_gz_path)

    return tar_gz_path


def local_preprocess(output_dir, dataset):
  """Preprocess data locally with Pandas

  Produce analysis used by training.

  Args:
    output_dir: The output directory to use.
    dataset: only CsvDataSet is supported currently.
  """
  import datalab.mlalpha as mlalpha
  if not isinstance(dataset, mlalpha.CsvDataSet):
    raise ValueError('Only CsvDataSet is supported')

  if len(dataset.input_files) != 1:
    raise ValueError('CsvDataSet should be built with a file pattern, not a '
                     'list of files.')

  # Write schema to a file.
  tmp_dir = tempfile.mkdtemp()
  _, schema_file_path = tempfile.mkstemp(dir=tmp_dir, suffix='.json',
                                        prefix='schema')
  try:
    file_io.write_string_to_file(schema_file_path, json.dumps(dataset.schema))

    args = ['local_preprocess',
            '--input_file_pattern=%s' % dataset.input_files[0],
            '--output_dir=%s' % output_dir,
            '--schema_file=%s' % schema_file_path]

    print('Starting local preprocessing.')
    preprocess.local_preprocess.main(args)
    print('Local preprocessing done.')
  finally:
    shutil.rmtree(tmp_dir)

def cloud_preprocess(output_dir, dataset, project_id=None):
  """Preprocess data in the cloud with BigQuery.

  Produce analysis used by training. This can take a while, even for small
  datasets. For small datasets, it may be faster to use local_preprocess.

  Args:
    output_dir: The output directory to use.
    dataset: only CsvDataSet is supported currently.
    project_id: project id the table is in. If none, uses the default project.
  """
  import datalab.mlalpha as mlalpha
  if not isinstance(dataset, mlalpha.CsvDataSet):
    raise ValueError('Only CsvDataSet is supported')

  if len(dataset.input_files) != 1:
    raise ValueError('CsvDataSet should be built with a file pattern, not a '
                     'list of files.')

  _assert_gcs_files([output_dir, dataset.input_files[0]])

  # Write schema to a file.
  tmp_dir = tempfile.mkdtemp()
  _, schema_file_path = tempfile.mkstemp(dir=tmp_dir, suffix='.json',
                                        prefix='schema')
  try:
    file_io.write_string_to_file(schema_file_path, json.dumps(dataset.schema))

    args = ['cloud_preprocess',
            '--input_file_pattern=%s' % dataset.input_files[0],
            '--output_dir=%s' % output_dir,
            '--schema_file=%s' % schema_file_path]


    print('Starting cloud preprocessing.')
    print('Track BigQuery status at')
    print('https://bigquery.cloud.google.com/queries/%s' % datalab_project_id())
    preprocess.cloud_preprocess.main(args)
    print('Cloud preprocessing done.')
  finally:
    shutil.rmtree(tmp_dir)


def local_train(train_dataset,
                eval_dataset,
                preprocess_output_dir,
                output_dir,
                transforms,
                model_type,
                max_steps=5000,
                num_epochs=None,
                train_batch_size=100,
                eval_batch_size=100,
                min_eval_frequency=100,
                top_n=None,
                layer_sizes=None,
                learning_rate=0.01,
                epsilon=0.0005):
  """Train model locally.
  Args:
    train_dataset: CsvDataSet
    eval_dataset: CsvDataSet
    preprocess_output_dir:  The output directory from preprocessing
    output_dir:  Output directory of training.
    transforms: file path or transform object. Example:
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

           For categorical colums, the transform supported depends on if the
           model is a linear or DNN model because tf.layers is uesed.
           For a linear model, the transforms supported are:
           i) {"transform": "sparse"}: Makes a sparse vector using the full
              vocabulary associated with the column (default).
           ii) {"transform": "hash_sparse", "hash_bucket_size": n}: First each
              string is hashed to an integer in the range [0, n), and then a
              sparse vector is used.

          For a DNN model, the categorical transforms that are supported are:
          i) {"transform": "one_hot"}: A one-hot vector using the full
              vocabulary is used. (default)
          ii) {"transform": "embedding", "embedding_dim": d}: Each label is
              embedded into an d-dimensional space.
          iii) {"transform": "hash_one_hot", "hash_bucket_size": n}: The label
              is first hashed into the range [0, n) and then a one-hot encoding
              is made.
          iv) {"transform": "hash_embedding", "hash_bucket_size": n,
               "embedding_dim": d}: First each label is hashed to [0, n), and
               then each integer is embedded into a d-dimensional space.
    model_type: One of linear_classification, linear_regression,
        dnn_classification, dnn_regression.
    max_steps: Int. Number of training steps to perform.
    num_epochs: Maximum number of training data epochs on which to train.
        The training job will run for max_steps or num_epochs, whichever occurs
        first.
    train_batch_size: number of rows to train on in one step.
    eval_batch_size: number of rows to eval in one step.
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
  """
  if len(train_dataset.input_files) != 1 or len(eval_dataset.input_files) != 1:
    raise ValueError('CsvDataSets must be built with a file pattern, not list '
                     'of files.')

  if file_io.file_exists(output_dir):
    raise ValueError('output_dir already exist. Use a new output path.')

  if isinstance(transforms, dict):
    # Make a transforms file.
    if not file_io.file_exists(output_dir):
      file_io.recursive_create_dir(output_dir)
    transforms_file = os.path.join(output_dir, 'transforms_file.json')
    file_io.write_string_to_file(
        transforms_file,
        json.dumps(transforms))
  else:
    transforms_file = transforms

  args = ['local_train',
          '--train_data_paths=%s' % train_dataset.input_files[0],
          '--eval_data_paths=%s' % eval_dataset.input_files[0],
          '--output_path=%s' % output_dir,
          '--preprocess_output_dir=%s' % preprocess_output_dir,
          '--transforms_file=%s' % transforms_file,
          '--model_type=%s' % model_type,
          '--max_steps=%s' % str(max_steps),
          '--train_batch_size=%s' % str(train_batch_size),
          '--eval_batch_size=%s' % str(eval_batch_size),
          '--min_eval_frequency=%s' % str(min_eval_frequency),
          '--learning_rate=%s' % str(learning_rate),
          '--epsilon=%s' % str(epsilon)]
  if num_epochs:
    args.append('--num_epochs=%s' % str(num_epochs))
  if top_n:
    args.append('--top_n=%s' % str(top_n))
  if layer_sizes:
    for i in range(len(layer_sizes)):
      args.append('--layer_size%s=%s' % (i+1, str(layer_sizes[i])))

  stderr = sys.stderr
  sys.stderr = sys.stdout
  print('Starting local training.')
  trainer.task.main(args)
  print('Local training done.')
  sys.stderr = stderr

def cloud_train(train_dataset,
                eval_dataset,
                preprocess_output_dir,
                output_dir,
                transforms,
                model_type,
                cloud_training_config,
                max_steps=5000,
                num_epochs=None,
                train_batch_size=100,
                eval_batch_size=100,
                min_eval_frequency=100,
                top_n=None,
                layer_sizes=None,
                learning_rate=0.01,
                epsilon=0.0005,
                job_name=None):
  """Train model using CloudML.

  See local_train() for a description of the args.
  Args:
    cloud_training_config: A CloudTrainingConfig object.
    job_name: Training job name. A default will be picked if None.
  """
  import datalab

  if len(train_dataset.input_files) != 1 or len(eval_dataset.input_files) != 1:
    raise ValueError('CsvDataSets must be built with a file pattern, not list '
                     'of files.')

  if file_io.file_exists(output_dir):
    raise ValueError('output_dir already exist. Use a new output path.')

  if isinstance(transforms, dict):
    # Make a transforms file.
    if not file_io.file_exists(output_dir):
      file_io.recursive_create_dir(output_dir)
    transforms_file = os.path.join(output_dir, 'transforms_file.json')
    file_io.write_string_to_file(
        transforms_file,
        json.dumps(transforms))
  else:
    transforms_file = transforms

  _assert_gcs_files([output_dir, train_dataset.input_files[0],
      eval_dataset.input_files[0], transforms_file,
      preprocess_output_dir])

  args = ['--train_data_paths=%s' % train_dataset.input_files[0],
          '--eval_data_paths=%s' % eval_dataset.input_files[0],
          '--output_path=%s' % output_dir,
          '--preprocess_output_dir=%s' % preprocess_output_dir,
          '--transforms_file=%s' % transforms_file,
          '--model_type=%s' % model_type,
          '--max_steps=%s' % str(max_steps),
          '--train_batch_size=%s' % str(train_batch_size),
          '--eval_batch_size=%s' % str(eval_batch_size),
          '--min_eval_frequency=%s' % str(min_eval_frequency),
          '--learning_rate=%s' % str(learning_rate),
          '--epsilon=%s' % str(epsilon)]
  if num_epochs:
    args.append('--num_epochs=%s' % str(num_epochs))
  if top_n:
    args.append('--top_n=%s' % str(top_n))
  if layer_sizes:
    for i in range(len(layer_sizes)):
      args.append('--layer_size%s=%s' % (i+1, str(layer_sizes[i])))

  job_request = {
    'package_uris': [_package_to_staging(output_dir)],
    'python_module': 'datalab_solutions.structured_data.trainer.task',
    'args': args
  }
  job_request.update(dict(cloud_training_config._asdict()))

  if not job_name:
    job_name = 'structured_data_train_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')
  job = datalab.mlalpha.Job.submit_training(job_request, job_name)
  print('Job request send. View status of job at')
  print('https://console.developers.google.com/ml/jobs?project=%s' %
        _default_project())

  return job


def local_predict(training_ouput_dir, data):
  """Runs local prediction on the prediction graph.

  Runs local prediction and returns the result in a Pandas DataFrame. For
  running prediction on a large dataset or saving the results, run
  local_batch_prediction or batch_prediction. Input data should fully match
  the schema that was used at training, except the target column should not
  exist.

  Args:
    training_ouput_dir: local path to the trained output folder.
    data: List of csv strings or a Pandas DataFrame that match the model schema.

  """
  # Save the instances to a file, call local batch prediction, and print it back
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

    model_dir = os.path.join(training_ouput_dir, 'model')
    if not file_io.file_exists(model_dir):
      raise ValueError('training_ouput_dir should contain the folder model')

    cmd = ['predict.py',
           '--predict_data=%s' % input_file_path,
           '--trained_model_dir=%s' % model_dir,
           '--output_dir=%s' % tmp_dir,
           '--output_format=csv',
           '--batch_size=100',
           '--mode=prediction',
           '--no-shard_files']

    print('Starting local prediction.')
    predict.predict.main(cmd)
    print('Local prediction done.')

    # Read the header file.
    schema_file = os.path.join(tmp_dir, 'csv_header.json')
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
      --origin gs://BUCKET/training_output_dir/model
  or these datalab commands:
  1) import datalab
     model = datalab.mlalpha.ModelVersions(MODEL_NAME)
     model.deploy(version_name=VERSION,
                  path='gs://BUCKET/training_output_dir/model')
  Note that the model must be on GCS.
  """
  import datalab.mlalpha as mlalpha


  if isinstance(data, pd.DataFrame):
    # write the df to csv.
    string_buffer = StringIO.StringIO()
    data.to_csv(string_buffer, header=None, index=False)
    input_data = string_buffer.getvalue().split('\n')

    #remove empty strings
    input_data = [line for line in input_data if line]
  else:
    input_data = data

  predictions = mlalpha.ModelVersions(model_name).predict(model_version, input_data)

  # Convert predictions into a dataframe
  df = pd.DataFrame(columns=sorted(predictions[0].keys()))
  for i in range(len(predictions)):
    for k, v in predictions[i].iteritems():
      df.loc[i, k] = v
  return df


def local_batch_predict(training_ouput_dir, prediction_input_file, output_dir,
                        mode,
                        batch_size=1000, shard_files=True, output_format='csv'):
  """Local batch prediction.

  Args:
    training_ouput_dir: The output folder of training.
    prediction_input_file: csv file pattern to a local file.
    output_dir: output location to save the results.
    mode: 'evaluation' or 'prediction'. If 'evaluation', the input data must
        contain a target column. If 'prediction', the input data must not
        contain a target column.
    batch_size: Int. How many instances to run in memory at once. Larger values
        mean better performace but more memeory consumed.
    shard_files: If false, the output files are not shardded.
    output_format: csv or json. Json file are json-newlined.
  """

  if mode == 'evaluation':
    model_dir = os.path.join(training_ouput_dir, 'evaluation_model')
  elif mode == 'prediction':
    model_dir = os.path.join(training_ouput_dir, 'model')
  else:
    raise ValueError('mode must be evaluation or prediction')

  if not file_io.file_exists(model_dir):
    raise ValueError('Model folder %s does not exist' % model_dir)

  cmd = ['predict.py',
         '--predict_data=%s' % prediction_input_file,
         '--trained_model_dir=%s' % model_dir,
         '--output_dir=%s' % output_dir,
         '--output_format=%s' % output_format,
         '--batch_size=%s' % str(batch_size),
         '--shard_files' if shard_files else '--no-shard_files',
         '--has_target' if mode == 'evaluation' else '--no-has_target'
         ]

  print('Starting local batch prediction.')
  predict.predict.main(cmd)
  print('Local batch prediction done.')



def cloud_batch_predict(training_ouput_dir, prediction_input_file, output_dir,
                        mode,
                        batch_size=1000, shard_files=True, output_format='csv'):
  """Cloud batch prediction. Submitts a Dataflow job.

  See local_batch_predict() for a description of the args.
  """
  if mode == 'evaluation':
    model_dir = os.path.join(training_ouput_dir, 'evaluation_model')
  elif mode == 'prediction':
    model_dir = os.path.join(training_ouput_dir, 'model')
  else:
    raise ValueError('mode must be evaluation or prediction')

  if not file_io.file_exists(model_dir):
    raise ValueError('Model folder %s does not exist' % model_dir)

  _assert_gcs_files([training_ouput_dir, prediction_input_file,
      output_dir])

  cmd = ['predict.py',
         '--cloud',
         '--project_id=%s' % _default_project(),
         '--predict_data=%s' % prediction_input_file,
         '--trained_model_dir=%s' % model_dir,
         '--output_dir=%s' % output_dir,
         '--output_format=%s' % output_format,
         '--batch_size=%s' % str(batch_size),
         '--shard_files' if shard_files else '--no-shard_files',
         '--extra_package=%s' % _package_to_staging(output_dir)]

  print('Starting cloud batch prediction.')
  predict.predict.main(cmd)
  print('See above link for job status.')
