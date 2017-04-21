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


"""Local implementation for preprocessing, training and prediction for inception model.
"""


import datetime


from . import _model
from . import _trainer
from . import _util


class Local(object):
  """Class for local training, preprocessing and prediction."""

  @staticmethod
  def preprocess(train_dataset, output_dir, eval_dataset, checkpoint):
    """Preprocess data locally."""

    import apache_beam as beam
    from google.datalab.utils import LambdaJob
    from . import _preprocess

    if checkpoint is None:
      checkpoint = _util._DEFAULT_CHECKPOINT_GSURL
    job_id = ('preprocess-image-classification-' +
              datetime.datetime.now().strftime('%y%m%d-%H%M%S'))
    # Project is needed for bigquery data source, even in local run.
    options = {
        'project': _util.default_project(),
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    p = beam.Pipeline('DirectRunner', options=opts)
    _preprocess.configure_pipeline(p, train_dataset, eval_dataset, checkpoint, output_dir, job_id)
    job = LambdaJob(lambda: p.run().wait_until_finish(), job_id)
    return job

  @staticmethod
  def train(input_dir, batch_size, max_steps, output_dir, checkpoint):
    """Train model locally."""
    from google.datalab.utils import LambdaJob

    if checkpoint is None:
      checkpoint = _util._DEFAULT_CHECKPOINT_GSURL

    labels = _util.get_labels(input_dir)
    model = _model.Model(labels, 0.5, checkpoint)
    task_data = {'type': 'master', 'index': 0}
    task = type('TaskSpec', (object,), task_data)
    job = LambdaJob(lambda: _trainer.Trainer(input_dir, batch_size, max_steps, output_dir,
                                             model, None, task).run_training(), 'training')
    return job

  @staticmethod
  def predict(model_dir, image_files, resize, show_image):
    """Predict using an model in a local or GCS directory."""

    from . import _predictor

    images = _util.load_images(image_files, resize=resize)
    labels_and_scores = _predictor.predict(model_dir, images)
    results = zip(image_files, images, labels_and_scores)
    ret = _util.process_prediction_results(results, show_image)
    return ret

  @staticmethod
  def batch_predict(dataset, model_dir, output_csv, output_bq_table):
    """Batch predict running locally."""

    import apache_beam as beam
    from google.datalab.utils import LambdaJob
    from . import _predictor

    if output_csv is None and output_bq_table is None:
      raise ValueError('output_csv and output_bq_table cannot both be None.')

    job_id = ('batch-predict-image-classification-' +
              datetime.datetime.now().strftime('%y%m%d-%H%M%S'))

    # Project is needed for bigquery data source, even in local run.
    options = {
        'project': _util.default_project(),
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    p = beam.Pipeline('DirectRunner', options=opts)
    _predictor.configure_pipeline(p, dataset, model_dir, output_csv, output_bq_table)
    job = LambdaJob(lambda: p.run().wait_until_finish(), job_id)
    return job
