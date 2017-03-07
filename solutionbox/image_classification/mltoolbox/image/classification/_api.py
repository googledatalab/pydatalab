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


"""Face functions for image classification.
"""


from . import _local
from . import _cloud


def preprocess_async(train_dataset, output_dir, eval_dataset=None, checkpoint=None, cloud=None):
  """Preprocess data. Produce output that can be used by training efficiently.

  Args:
    train_dataset: training data source to preprocess. Can be CsvDataset or BigQueryDataSet.
        If eval_dataset is None, the pipeline will randomly split train_dataset into
        train/eval set with 7:3 ratio.
    output_dir: The output directory to use. Preprocessing will create a sub directory under
        it for each run, and also update "latest" file which points to the latest preprocessed
        directory. Users are responsible for cleanup. Can be local or GCS path.
    eval_dataset: evaluation data source to preprocess. Can be CsvDataset or BigQueryDataSet.
        If specified, it will be used for evaluation during training, and train_dataset will be
        completely used for training.
    checkpoint: the Inception checkpoint to use. If None, a default checkpoint is used.
    cloud: a DataFlow pipeline option dictionary such as {'num_workers': 3}. If anything but
        not None, it will run in cloud. Otherwise, it runs locally.

  Returns:
    A google.datalab.utils.Job object that can be used to query state from or wait.
  """

  if cloud is None:
    return _local.Local.preprocess(train_dataset, output_dir, eval_dataset, checkpoint)

  if not isinstance(cloud, dict):
    cloud = {}
  return _cloud.Cloud.preprocess(train_dataset, output_dir, eval_dataset, checkpoint, cloud)


def preprocess(train_dataset, output_dir, eval_dataset=None, checkpoint=None, cloud=None):
  """Blocking version of preprocess_async(). The only difference is that it blocks the caller
     until the job finishes, and it does not have a return value.
  """
  job = preprocess_async(train_dataset, output_dir, eval_dataset, checkpoint, cloud)
  job.wait()
  print job.state


def train_async(input_dir, batch_size, max_steps, output_dir, checkpoint=None, cloud=None):
  """Train model. The output can be used for batch prediction or online deployment.

  Args:
    input_dir: A directory path containing preprocessed results. Can be local or GCS path.
    batch_size: size of batch used for training.
    max_steps: number of steps to train.
    output_dir: The output directory to use. Can be local or GCS path.
    checkpoint: the Inception checkpoint to use. If None, a default checkpoint is used.
    cloud: a google.datalab.ml.CloudTrainingConfig object to let it run in cloud.
        If None, it runs locally.

  Returns:
    A google.datalab.utils.Job object that can be used to query state from or wait.
  """

  if cloud is None:
    return _local.Local.train(input_dir, batch_size, max_steps, output_dir, checkpoint)

  return _cloud.Cloud.train(input_dir, batch_size, max_steps, output_dir, checkpoint, cloud)


def train(input_dir, batch_size, max_steps, output_dir, checkpoint=None, cloud=None):
  """Blocking version of train_async(). The only difference is that it blocks the caller
     until the job finishes, and it does not have a return value.
  """

  job = train_async(input_dir, batch_size, max_steps, output_dir, checkpoint, cloud)
  job.wait()
  print job.state


def predict(model, image_files, resize=False, show_image=True, cloud=None):
  """Predict using an model in a local or GCS directory (offline), or a deployed model (online).

  Args:
    model: if cloud is None, a local or GCS directory of a trained model. Otherwise, it specifies
        a deployed model identified by model.version, such as "imagemodel.v1".
    image_files: The paths to the image files to predict labels. Can be local or GCS paths.
    resize: Whether to resize the image to a reasonable size (300x300) before prediction.
    show_image: Whether to show images in the results.
    cloud: if None, predicts with offline model locally. Otherwise, predict with a deployed online model.

  Returns:
    A pandas DataFrame including the prediction results.
  """

  print('Predicting...')
  if cloud is None:
    results = _local.Local.predict(model, image_files, resize, show_image)
  else:
    results = _cloud.Cloud.predict(model, image_files, resize, show_image)
  return results


def batch_predict_async(dataset, model_dir, output_csv=None, output_bq_table=None, cloud=None):
  """Batch prediction with an offline model.

  Args:
    dataset: CsvDataSet or BigQueryDataSet for batch prediction input. Can contain either
        one column 'image_url', or two columns with another being 'label'.
    model_dir: The directory of a trained inception model. Can be local or GCS paths.
    output_csv: The output csv file for prediction results. If specified,
        it will also output a csv schema file with the name output_csv + '.schema.json'.
    output_bq_table: if specified, the output BigQuery table for prediction results.
        output_csv and output_bq_table can both be set.
    cloud: a DataFlow pipeline option dictionary such as {'num_workers': 3}. If anything but
        not None, it will run in cloud. Otherwise, it runs locally.
        If specified, it must include 'temp_location' with value being a GCS path, because cloud
        run requires a staging GCS directory.

  Raises:
    ValueError if both output_csv and output_bq_table are None, or if cloud is not None
        but it does not include 'temp_location'.

  Returns:
    A google.datalab.utils.Job object that can be used to query state from or wait.
  """
  if cloud is None:
    return _local.Local.batch_predict(dataset, model_dir, output_csv, output_bq_table)

  if not isinstance(cloud, dict):
    cloud = {}
  return _cloud.Cloud.batch_predict(dataset, model_dir, output_csv, output_bq_table, cloud)


def batch_predict(dataset, model_dir, output_csv=None, output_bq_table=None, cloud=None):
  """Blocking version of batch_predict_async(). The only difference is that it blocks the caller
     until the job finishes, and it does not have a return value.
  """

  job = batch_predict_async(dataset, model_dir, output_csv, output_bq_table, cloud)
  job.wait()
  print job.state
