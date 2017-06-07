from __future__ import absolute_import

import base64
import json
import logging
import os
from PIL import Image
import random
import six
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid

import tensorflow as tf
from tensorflow.python.lib.io import file_io

CODE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'mltoolbox', 'code_free_ml', 'data'))


class TestCloudServicesTrainer(unittest.TestCase):
  """Tests everything using the cloud services.

  Run cloud analyze, cloud transformation, cloud training, and cloud batch
  prediction. Easy step is done by making a subprocess call to python or
  gcloud.

  Each step has a local 'cloud' variable that can be mannually set to False to
  run the local version of the step. This is usefull when debugging as not
  every step needs to use cloud services.

  Because of the cloud overhead, this test easily takes ~40 mins to finish.

  Test files will be uploaded into a new bucket named temp_pydatalab_test_*
  using the default project from gcloud. The bucket is removed at the end
  of the test.
  """
  def __init__(self, *args, **kwargs):
    super(TestCloudServicesTrainer, self).__init__(*args, **kwargs)

    self._max_steps = 2000

    # Log everything
    self._logger = logging.getLogger('TestStructuredDataLogger')
    self._logger.setLevel(logging.DEBUG)
    if not self._logger.handlers:
      self._logger.addHandler(logging.StreamHandler(stream=sys.stdout))

  def setUp(self):
    random.seed(12321)
    self._local_dir = tempfile.mkdtemp()  # Local folder for temp files.
    self._gs_dir = 'gs://temp_pydatalab_test_264bf887749047e0a09b35f5ebf672be'
    #self._gs_dir = 'gs://temp_pydatalab_test_%s' % uuid.uuid4().hex
    #subprocess.check_call('gsutil mb %s' % self._gs_dir, shell=True)

    self._input_files = os.path.join(self._gs_dir, 'input_files')

    self._analysis_output = os.path.join(self._gs_dir, 'analysis_output')
    self._transform_output = os.path.join(self._gs_dir, 'transform_output')
    self._train_output = os.path.join(self._gs_dir, 'train_output')
    self._prediction_output = os.path.join(self._gs_dir, 'prediction_output')

    file_io.recursive_create_dir(self._input_files)

    self._csv_train_filename = os.path.join(self._input_files, 'train_csv_data.csv')
    self._csv_eval_filename = os.path.join(self._input_files, 'eval_csv_data.csv')
    self._csv_predict_filename = os.path.join(self._input_files, 'predict_csv_data.csv')
    self._schema_filename = os.path.join(self._input_files, 'schema_file.json')
    self._features_filename = os.path.join(self._input_files, 'features_file.json')

    self._image_files = None

  def tearDown(self):
    self._logger.debug('TestCloudServicesTrainer: removing folders %s, %s' % (self._local_dir, self._gs_dir))
    shutil.rmtree(self._local_dir)
    #subprocess.check_call('gsutil -m rm -r %s' % self._gs_dir, shell=True)

  def _make_image_files(self):
    """Makes random images and uploads them to GCS.

    The images are first made locally and then moved to GCS for speed.
    """
    self._image_files = []

    for i in range(10):
      r = random.randint(0, 255)
      g = random.randint(0, 255)
      b = random.randint(0, 255)
      img_name = 'img%02d.jpg' % i
      local_img = os.path.join(self._local_dir, img_name)
      img = Image.new('RGBA', size=(300, 300), color=(155, 0, 0))
      img.save(local_img)

      self._image_files.append((r, g, b, os.path.join(self._input_files, img_name)))

    cmd = 'gsutil -m mv %s/img*.jpg %s/' % (self._local_dir, self._input_files)
    subprocess.check_call(cmd, shell=True)

  def _make_csv_data(self, filename, num_rows, keep_target=True, embedded_image=False):
    """Writes csv data.

    Builds a linear model that uses 1 numerical column and an image column.

    Args:
      filename: gcs filepath
      num_rows: how many rows of data will be generated.
      keep_target: if false, the target column is missing.
      embedded_image: if true, the image column will be the base64 data
    """
    def _drop_out(x):
      # Make 5% of the data missing
      if random.uniform(0, 1) < 0.05:
        return ''
      return x

    local_file = os.path.join(self._local_dir, 'data.csv')
    with open(local_file, 'w') as f:
      for i in range(num_rows):
        num = random.randint(0, 20)
        r, g, b, img_path = random.choice(self._image_files)

        if embedded_image:
          with file_io.FileIO(img_path, 'r') as img_file:
            img_bytes = Image.open(img_file)
          buf = six.StringIO()
          img_bytes.save(buf, 'JPEG')
          img_data = base64.urlsafe_b64encode(buf.getvalue())
        else:
          img_data = img_path

        # Build a simple linear model
        t = -10 + 0.5 * num + 0.1 * r

        num = _drop_out(num)
        if num is not '':  # Don't drop every column
          img_data = _drop_out(img_data)

        if keep_target:
          csv_line = "{key},{target},{num},{img_data}\n".format(
              key=i,
              target=t,
              num=num,
              img_data=img_data)
        else:
          csv_line = "{key},{num},{img_data}\n".format(
              key=i,
              num=num,
              img_data=img_data)

        f.write(csv_line)
    subprocess.check_call('gsutil cp %s %s' % (local_file, filename), shell=True)

  def _get_default_project_id(self):
    with open(os.devnull, 'w') as dev_null:
      cmd = 'gcloud config list project --format=\'value(core.project)\''
      return subprocess.check_output(cmd, shell=True, stderr=dev_null).strip()

  def _run_analyze(self):
    """Runs analysis using BigQuery from csv files."""

    cloud = True
    self._logger.debug('Create input files')

    features = {
        'num': {'transform': 'scale'},
        'img': {'transform': 'image_to_vec'},
        'target': {'transform': 'target'},
        'key': {'transform': 'key'}}
    file_io.write_string_to_file(self._features_filename, json.dumps(features, indent=2))

    schema = [
        {'name': 'key', 'type': 'integer'},
        {'name': 'target', 'type': 'float'},
        {'name': 'num', 'type': 'integer'},
        {'name': 'img', 'type': 'string'}]
    file_io.write_string_to_file(self._schema_filename, json.dumps(schema, indent=2))

    self._make_image_files()

    self._make_csv_data(self._csv_train_filename, 30, True, False)
    self._make_csv_data(self._csv_eval_filename, 10, True, False)
    self._make_csv_data(self._csv_predict_filename, 5, False, True)

    cmd = ['python %s' % os.path.join(CODE_PATH, 'analyze_data.py'),
           '--cloud' if cloud else '',
           '--output-dir=' + self._analysis_output,
           '--csv-file-pattern=' + self._csv_train_filename,
           '--csv-schema-file=' + self._schema_filename,
           '--features-file=' + self._features_filename]

    self._logger.debug('Running subprocess: %s \n\n' % ' '.join(cmd))
    subprocess.check_call(' '.join(cmd), shell=True)
    self.assertTrue(file_io.file_exists(os.path.join(self._analysis_output, 'stats.json')))
    self.assertTrue(file_io.file_exists(os.path.join(self._analysis_output, 'schema.json')))
    self.assertTrue(file_io.file_exists(os.path.join(self._analysis_output, 'features.json')))

  def _run_transform(self):
    """Runs DataFlow for makint tf.example files.

    Only the train file uses DataFlow, the eval file runs beam locally to save
    time.
    """
    cloud = True
    extra_args = []
    if cloud:
      extra_args = ['--cloud',
                    '--job-name=test-mltoolbox-df-%s' % uuid.uuid4().hex,
                    '--project-id=%s' % self._get_default_project_id(),
                    '--num-workers=3']

    cmd = ['python %s' % os.path.join(CODE_PATH, 'transform_raw_data.py'),
           '--csv-file-pattern=' + self._csv_train_filename,
           '--analysis-output-dir=' + self._analysis_output,
           '--output-filename-prefix=features_train',
           '--output-dir=' + self._transform_output,
           '--shuffle'] + extra_args

    self._logger.debug('Running subprocess: %s \n\n' % ' '.join(cmd))
    subprocess.check_call(' '.join(cmd), shell=True)

    # Don't wate time running a 2nd DF job, run it locally.
    cmd = ['python %s' % os.path.join(CODE_PATH, 'transform_raw_data.py'),
           '--csv-file-pattern=' + self._csv_eval_filename,
           '--analysis-output-dir=' + self._analysis_output,
           '--output-filename-prefix=features_eval',
           '--output-dir=' + self._transform_output]

    self._logger.debug('Running subprocess: %s \n\n' % ' '.join(cmd))
    subprocess.check_call(' '.join(cmd), shell=True)

    # Check the files were made
    train_files = file_io.get_matching_files(
        os.path.join(self._transform_output, 'features_train*'))
    eval_files = file_io.get_matching_files(
        os.path.join(self._transform_output, 'features_eval*'))
    self.assertNotEqual([], train_files)
    self.assertNotEqual([], eval_files)

  def _run_training_transform(self):
    """Runs training starting with transformed tf.example files."""

    cloud = True
    if cloud:
      cmd = ['gcloud ml-engine jobs submit training test_mltoolbox_train_%s' % uuid.uuid4().hex,
             '--runtime-version=1.0',
             '--scale-tier=STANDARD_1',
             '--stream-logs']
    else:
      cmd = ['gcloud ml-engine local train']

    cmd = cmd + [
        '--module-name trainer.task',
        '--job-dir=' + self._train_output,
        '--package-path=' + os.path.join(CODE_PATH, 'trainer'),
        '--',
        '--train-data-paths=' + os.path.join(self._transform_output, 'features_train*'),
        '--eval-data-paths=' + os.path.join(self._transform_output, 'features_eval*'),
        '--analysis-output-dir=' + self._analysis_output,
        '--model-type=linear_regression',
        '--train-batch-size=10',
        '--eval-batch-size=10',
        '--max-steps=' + str(self._max_steps)]

    self._logger.debug('Running subprocess: %s \n\n' % ' '.join(cmd))
    subprocess.check_call(' '.join(cmd), shell=True)

    # Check the saved model was made.
    self.assertTrue(file_io.file_exists(
        os.path.join(self._train_output, 'model', 'saved_model.pb')))
    self.assertTrue(file_io.file_exists(
        os.path.join(self._train_output, 'evaluation_model', 'saved_model.pb')))

  def _run_batch_prediction(self):
    """Run batch prediction using the cloudml engine prediction service.

    There is no local version of this step as it's the last step.
    """

    job_name = 'test_mltoolbox_batchprediction_%s' % uuid.uuid4().hex
    cmd = ['gcloud ml-engine jobs submit prediction ' + job_name,
           '--data-format=TEXT',
           '--input-paths=' + self._csv_predict_filename,
           '--output-path=' + self._prediction_output,
           '--model-dir=' + os.path.join(self._train_output, 'model'),
           '--runtime-version=1.0',
           '--region=us-central1']
    self._logger.debug('Running subprocess: %s \n\n' % ' '.join(cmd))
    subprocess.check_call(' '.join(cmd), shell=True)  # async call.
    subprocess.check_call('gcloud ml-engine jobs stream-logs ' + job_name, shell=True)

  def test_cloud_workflow(self):
    #self._run_analyze()
    #self._run_transform()
    self._run_training_transform()
    self._run_batch_prediction()


if __name__ == '__main__':
    unittest.main()
