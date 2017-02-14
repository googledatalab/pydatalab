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
import yaml

from . import preprocess
from . import trainer
from . import predict

#_SETUP_PY = '/datalab/packages_setup/structured_data/setup.py'
#_TF_VERSION = 'tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl'
#_TF_WHL = '/datalab/packages_setup/structured_data'


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
        os.path.join(os.path.dirname(__file__), 'setup.py'))
    tar_gz_path = os.path.join(staging_package_url, 'staging', 'sd.tar.gz')

    print('Building package in %s and uploading to %s' % 
          (package_root, tar_gz_path))
    mlalpha.package_and_copy(package_root, setup_path, tar_gz_path)


    return tar_gz_path


def local_preprocess(output_dir, input_file_pattern, schema_file):
  """Preprocess data locally with Pandas

  Produce analysis used by training.

  Args:
    output_dir: The output directory to use.
    input_file_pattern: String. File pattern what will expand into a list of csv
        files.
    schema_file: File path to the schema file.
    
  """
  args = ['local_preprocess',
          '--input_file_pattern=%s' % input_file_pattern,
          '--output_dir=%s' % output_dir,
          '--schema_file=%s' % schema_file]

  print('Starting local preprocessing.')
  preprocess.local_preprocess.main(args)
  print('Local preprocessing done.')

def cloud_preprocess(output_dir, input_file_pattern=None, schema_file=None, bigquery_table=None, project_id=None):
  """Preprocess data in the cloud with BigQuery.

  Produce analysis used by training. This can take a while, even for small 
  datasets. For small datasets, it may be faster to use local_preprocess.

  Args:
    output_dir: The output directory to use.
    input_file_path: String. File pattern what will expand into a list of csv
        files.
    schema_file: File path to the schema file.
    bigquery_table: bigquery name in the form 'dataset.tabele_name'
    project_id: project id the table is in. If none, uses the default project.
  """
  _assert_gcs_files([output_dir, input_file_pattern])

  args = ['cloud_preprocess',
          '--output_dir=%s' % output_dir]

  if input_file_pattern:
    args.append('--input_file_pattern=%s' % input_file_pattern)
  if schema_file:
    args.append('--schema_file=%s' % schema_file)
  if not project_id:
    project_id = _default_project()
  if bigquery_table:
    full_name = project_id + ':' + bigquery_table
    args.append('--bigquery_table=%s' % full_name)
  
  print('Starting cloud preprocessing.')
  print('Track BigQuery status at')
  print('https://bigquery.cloud.google.com/queries/%s' % project_id)
  preprocess.cloud_preprocess.main(args)
  print('Cloud preprocessing done.')


def local_train(train_file_pattern,
                eval_file_pattern, 
                preprocess_output_dir, 
                output_dir,
                transforms_file,
                model_type,
                max_steps,
                top_n=None,
                layer_sizes=None):
  """Train model locally.
  Args:
    train_file_pattern: train csv file
    eval_file_pattern: eval csv file
    preprocess_output_dir:  The output directory from preprocessing
    output_dir:  Output directory of training.
    transforms_file: File path to the transforms file. Example
        {
          "col_A": {"transform": "scale", "default": 0.0},
          "col_B": {"transform": "scale","value": 4},
          "col_D": {"transform": "hash_one_hot", "hash_bucket_size": 4},
          "col_target": {"transform": "target"},
          "col_key": {"transform": "key"}
        }
        The keys correspond to the columns in the input files as defined by the
        schema file during preprocessing. Some notes
        1) The "key" transform is required, but the "target" transform is 
           optional, as the target column must be the first column in the input
           data, and all other transfroms are optional.
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
           model is a linear or DNN model. For a linear model, the transforms
           supported are:
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
    top_n: Int. For classification problems, the output graph will contain the
        labels and scores for the top n classes with a default of n=1. Use 
        None for regression problems.
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
  for i in range(len(layer_sizes)):
    args.append('--layer_size%s=%s' % (i+1, str(layer_sizes[i])))
  if top_n:
    args.append('--top_n=%s' % str(top_n))

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
                top_n=None,
                layer_sizes=None,
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
    transforms_file: File path to the transforms file. See local_train for 
        a long description of this file.
    model_type: One of linear_classification, linear_regression,
        dnn_classification, dnn_regression.
    max_steps: Int. Number of training steps to perform.
    top_n: Int. For classification problems, the output graph will contain the
        labels and scores for the top n classes with a default of n=1.
        Use None for regression problems.
    layer_sizes: List. Represents the layers in the connected DNN.
        If the model type is DNN, this must be set. Example [10, 3, 2], this 
        will create three DNN layers where the first layer will have 10 nodes, 
        the middle layer will have 3 nodes, and the laster layer will have 2
        nodes.
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
                     preprocess_output_dir, transforms_file, output_dir])

  # TODO: Convert args to a dictionary so we can use datalab's cloudml trainer.
  args = ['--train_data_paths=%s' % train_file_pattern,
          '--eval_data_paths=%s' % eval_file_pattern,
          '--output_path=%s' % output_dir,
          '--preprocess_output_dir=%s' % preprocess_output_dir,
          '--transforms_file=%s' % transforms_file,
          '--model_type=%s' % model_type,
          '--max_steps=%s' % str(max_steps)]
  for i in range(len(layer_sizes)):
    args.append('--layer_size%s=%s' % (i+1, str(layer_sizes[i])))
  if top_n:
    args.append('--top_n=%s' % str(top_n))    

  job_request = {
    'package_uris': [_package_to_staging(output_dir)],
    'python_module': 'datalab_solutions.structured_data.trainer.task',
    'scale_tier': scale_tier,
    'region': region,
    'args': args
  }
  # Local import because cloudml service does not have datalab
  import datalab
  cloud_runner = datalab.mlalpha.CloudRunner(job_request)
  if not job_name:
    job_name = 'structured_data_train_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')
  job = datalab.mlalpha.Job.submit_training(job_request, job_name)
  
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


def local_predict(model_dir, data):
  """Runs local prediction.

  Runs local prediction and returns the result in a Pandas DataFrame. For
  running prediction on a large dataset or saving the results, run
  local_batch_prediction or batch_prediction.

  Args:
    model_dir: local path to the trained mode. Usually, this is 
        training_output_dir/model.
    data: List of csv strings that match the model schema. Or a pandas DataFrame 
        where the columns match the model schema. The first column,
        the target column, could be missing.
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


    cmd = ['predict.py',
           '--predict_data=%s' % input_file_path,
           '--trained_model_dir=%s' % model_dir,
           '--output_dir=%s' % tmp_dir,
           '--output_format=csv',
           '--batch_size=100',
           '--no-shard_files']

    print('Starting local prediction.')
    predict.predict.main(cmd)
    print('Local prediction done.')
    
    # Read the header file.
    header_file = os.path.join(tmp_dir, 'csv_header.txt')
    with open(header_file, 'r') as f:
      header = f.readline()

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
                              names=header.split(','))
    return predictions
  finally:
    shutil.rmtree(tmp_dir)


def cloud_predict(model_name, model_version, data, is_target_missing=False):
  """Use Online prediction.

  Runs online prediction in the cloud and prints the results to the screen. For
  running prediction on a large dataset or saving the results, run
  local_batch_prediction or batch_prediction.

  Args:
    model_name: deployed model name
    model_verion: depoyed model version
    data: List of csv strings that match the model schema. Or a pandas DataFrame 
        where the columns match the model schema. The first column,
        the target column, is assumed to exist in the data. 
    is_target_missing: If true, prepends a ',' in each csv string or adds an 
        empty DataFrame column. If the csv data has a leading ',' keep this flag
        False. Example: 
        1) If data = ['target,input1,input2'], then set is_target_missing=False.
        2) If data = [',input1,input2'], then set is_target_missing=False.
        3) If data = ['input1,input2'], then set is_target_missing=True.

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
  import datalab.mlalpha as mlalpha 


  if isinstance(data, pd.DataFrame):
    # write the df to csv.
    string_buffer = StringIO.StringIO()
    data.to_csv(string_buffer, header=None, index=False)
    csv_lines = string_buffer.getvalue().split('\n')

    if is_target_missing:
      input_data = [',' + csv for csv in csv_lines]
    else:
      input_data = csv_lines
  else:
    if is_target_missing:
      input_data = [ ',' + csv for csv in data]
    else:
      input_data = data

  predictions = mlalpha.ModelVersions(model_name).predict(model_version, input_data)

  # Convert predictions into a dataframe
  df = pd.DataFrame(columns=sorted(predictions[0].keys()))
  for i in range(len(predictions)):
    for k, v in predictions[i].iteritems():
      df.loc[i, k] = v
  return df


def local_batch_predict(model_dir, prediction_input_file, output_dir,
                        batch_size=1000, shard_files=True, output_format='csv'):
  """Local batch prediction.

  Args:
    model_dir: local file path to trained model. Usually, this is 
        training_output_dir/model.
    prediction_input_file: csv file pattern to a local file.
    output_dir: local output location to save the results.
    batch_size: Int. How many instances to run in memory at once. Larger values
        mean better performace but more memeory consumed.
    shard_files: If false, the output files are not shardded.
    output_format: csv or json. Json file are json-newlined.
  """
  cmd = ['predict.py',
         '--predict_data=%s' % prediction_input_file,
         '--trained_model_dir=%s' % model_dir,
         '--output_dir=%s' % output_dir,
         '--output_format=%s' % output_format,
         '--batch_size=%s' % str(batch_size)]

  if shard_files:
    cmd.append('--shard_files')
  else:
    cmd.append('--no-shard_files')

  print('Starting local batch prediction.')
  predict.predict.main(cmd)
  print('Local batch prediction done.')



def cloud_batch_predict(model_dir, prediction_input_file, output_dir,
                        batch_size=1000, shard_files=True, output_format='csv'):
  """Cloud batch prediction. Submitts a Dataflow job.

  Args:
    model_dir: GSC file path to trained model. Usually, this is 
        training_output_dir/model.
    prediction_input_file: csv file pattern to a GSC file.
    output_dir: Location to save the results on GCS.
    batch_size: Int. How many instances to run in memory at once. Larger values
        mean better performace but more memeory consumed.
    shard_files: If false, the output files are not shardded.
    output_format: csv or json. Json file are json-newlined.
  """
  cmd = ['predict.py',
         '--cloud',
         '--project_id=%s' % _default_project(),
         '--predict_data=%s' % prediction_input_file,
         '--trained_model_dir=%s' % model_dir,
         '--output_dir=%s' % output_dir,
         '--output_format=%s' % output_format,
         '--batch_size=%s' % str(batch_size),
         '--extra_package=%s' % _package_to_staging(output_dir)]
  print(cmd)

  if shard_files:
    cmd.append('--shard_files')
  else:
    cmd.append('--no-shard_files')

  print('Starting cloud batch prediction.')
  predict.predict.main(cmd)
  print('See above link for job status.')
