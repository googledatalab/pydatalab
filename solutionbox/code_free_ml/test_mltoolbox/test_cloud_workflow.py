from __future__ import absolute_import

import base64
import json
import logging
import os
from PIL import Image
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid

import tensorflow as tf
from tensorflow.python.lib.io import file_io


class TestCloudServicesTrainer(unittest.TestCase):
  """Tests everything using the cloud
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
    self._local_dir = tempfile.mkdtemp()
    self._gs_dir = 'gs://temp_pydatalab_test_%s' % uuid.uuid4().hex
    subprocess.check_call('gsutil mb %s' % self._gs_dir, shell=True)

    self._input_files = os.path.join(self._gs_dir, 'input_files')

    self._analysis_output = os.path.join(self._gs_dir, 'analysis_output')
    self._transform_output = os.path.join(self._gs_dir, 'transform_output')
    self._train_output = os.path.join(self._gs_dir, 'train_output')

    file_io.recursive_create_dir(self._input_files)
    file_io.recursive_create_dir(self._analysis_output)
    file_io.recursive_create_dir(self._transform_output)
    file_io.recursive_create_dir(self._train_output)

    self._csv_train_filename = os.path.join(self._input_files, 'train_csv_data.csv')
    self._csv_eval_filename = os.path.join(self._input_files, 'eval_csv_data.csv')
    self._csv_predict_filename = os.path.join(self._input_files, 'predict_csv_data.csv')
    self._schema_filename = os.path.join(self._input_files, 'schema_file.json')
    self._features_filename = os.path.join(self._input_files, 'features_file.json')

    self._image_files = None

  def tearDown(self):
    self._logger.debug('TestCloudServicesTrainer: removing folders %s, %s' % (self._local_dir, self._gs_dir))
    shutil.rmtree(self._local_dir)
    subprocess.check_call('gsutil -m rm -r %s' % self._gs_dir, shell=True)

  def _make_image_files(self):
    self._image_files = []

    for i in range(50):
      r = random.randint(0, 255)
      g = random.randint(0, 255)
      b = random.randint(0, 255)
      img_name = 'img%02d.jpg' % i
      local_img = os.path.join(self._local_dir, img_name)
      img = Image.new('RGBA', size=(300, 300), color=(155, 0, 0))
      img.save(local_img)

      self._image_files.append((r, g, b, img_name))

    subprocess.check_call('gsutil -m mv %s/img*.jpg %s/' % (self._local_dir, self._input_files), shell=True)

  def _make_csv_data(self, filename, num_rows, keep_target=True):
    """Writes csv data.

    Builds a linear model.

    Args:
      filename: gcs filepath
      num_rows: how many rows of data will be generated.
      keep_target: if false, the target column is missing. 
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

        # Build a simple linear model
        t = -100 + 0.5 * num + r - g + b

        num = _drop_out(num)
        r = _drop_out(r)
        g = 

        if keep_target:
          csv_line = "{key},{target},{num},{img_path}\n".format(
              key=i,
              target=t,
              num=num,
              img_path=img_path)
        else:
          csv_line = "{key},{num},{img_path}\n".format(
              key=i,
              num=num,
              img_path=img_path)

        f.write(csv_line)
    subprocess.check_call('gsutil cp %s %s' % (local_file, filename), shell=True)

  def test_cloud_workflow(self):
    self._run_analyze()


  def _run_analyze(self, problem_type, with_image=False):

    features = {
        'num': {'transform': 'scale'},
        'img': {'transform': 'image_to_vec'},
        'target': {'transform': 'target'},
        'key': {'transform': 'key'}}

    schema = [
        {'name': 'key', 'type': 'integer'},
        {'name': 'target', 'type': 'float'},
        {'name': 'num', 'type': 'integer'},
        {'name': 'num_scale', 'type': 'float'},
        {'name': 'str_one_hot', 'type': 'string'},
        {'name': 'str_embedding', 'type': 'string'},
        {'name': 'str_bow', 'type': 'string'},
        {'name': 'str_tfidf', 'type': 'string'}]
    if with_image:
      schema.append({'name': 'image', 'type': 'string'})




    self._logger.debug('Create input files')
    self._make_image_files()

    self._make_csv_data(self._csv_train_filename, 5000, True)
    self._make_csv_data(self._csv_eval_filename, 500, True)
    self._make_csv_data(self._csv_predict_filename, 100, False)


    self._schema = schema

    file_io.write_string_to_file(self._schema_filename, json.dumps(schema, indent=2))
    file_io.write_string_to_file(self._features_filename, json.dumps(features, indent=2))

    if with_image:
      self.make_image_files()

    self.make_csv_data(self._csv_train_filename, 200, problem_type, True, with_image)
    self.make_csv_data(self._csv_eval_filename, 100, problem_type, True, with_image)
    self.make_csv_data(self._csv_predict_filename, 100, problem_type, False, with_image)

    cmd = ['python %s' % os.path.join(CODE_PATH, 'analyze_data.py'),
           '--output-dir=' + self._analysis_output,
           '--csv-file-pattern=' + self._csv_train_filename,
           '--csv-schema-file=' + self._schema_filename,
           '--features-file=' + self._features_filename]

    subprocess.check_call(' '.join(cmd), shell=True)

  def _run_transform(self):
    cmd = ['python %s' % os.path.join(CODE_PATH, 'transform_raw_data.py'),
           '--csv-file-pattern=' + self._csv_train_filename,
           '--analysis-output-dir=' + self._analysis_output,
           '--output-filename-prefix=features_train',
           '--output-dir=' + self._transform_output,
           '--shuffle']

    self._logger.debug('Running subprocess: %s \n\n' % ' '.join(cmd))
    subprocess.check_call(' '.join(cmd), shell=True)

    cmd = ['python %s' % os.path.join(CODE_PATH, 'transform_raw_data.py'),
           '--csv-file-pattern=' + self._csv_eval_filename,
           '--analysis-output-dir=' + self._analysis_output,
           '--output-filename-prefix=features_eval',
           '--output-dir=' + self._transform_output]

    self._logger.debug('Running subprocess: %s \n\n' % ' '.join(cmd))
    subprocess.check_call(' '.join(cmd), shell=True)

  def _run_training_transform(self, problem_type, model_type, extra_args=[]):
    """Runs training starting with transformed tf.example files.

    Args:
      problem_type: 'regression' or 'classification'
      model_type: 'linear' or 'dnn'
      extra_args: list of strings to pass to the trainer.
    """
    cmd = ['python %s' % os.path.join(CODE_PATH, 'trainer', 'task.py'),
           '--train-data-paths=' + os.path.join(self._transform_output, 'features_train*'),
           '--eval-data-paths=' + os.path.join(self._transform_output, 'features_eval*'),
           '--job-dir=' + self._train_output,
           '--analysis-output-dir=' + self._analysis_output,
           '--model-type=%s_%s' % (model_type, problem_type),
           '--train-batch-size=100',
           '--eval-batch-size=50',
           '--max-steps=' + str(self._max_steps)] + extra_args

    self._logger.debug('Running subprocess: %s \n\n' % ' '.join(cmd))
    subprocess.check_call(' '.join(cmd), shell=True)

  def _run_training_raw(self, problem_type, model_type, extra_args=[]):
    """Runs training starting from raw csv data.

    Args:
      problem_type: 'regression' or 'classification'
      model_type: 'linear' or 'dnn'
      extra_args: list of strings to pass to the trainer.
    """
    cmd = ['python %s' % os.path.join(CODE_PATH, 'trainer', 'task.py'),
           '--train-data-paths=' + self._csv_train_filename,
           '--eval-data-paths=' + self._csv_eval_filename,
           '--job-dir=' + self._train_output,
           '--analysis-output-dir=' + self._analysis_output,
           '--model-type=%s_%s' % (model_type, problem_type),
           '--train-batch-size=100',
           '--eval-batch-size=50',
           '--max-steps=' + str(self._max_steps),
           '--run-transforms'] + extra_args

    self._logger.debug('Running subprocess: %s \n\n' % ' '.join(cmd))
    subprocess.check_call(' '.join(cmd), shell=True)

  def _check_model(self, problem_type, model_type, with_image=False):
    """Checks that both exported prediction graphs work."""

    for has_target in [True, False]:
      if has_target:
        model_path = os.path.join(self._train_output, 'evaluation_model')
      else:
        model_path = os.path.join(self._train_output, 'model')

      self._logger.debug('Checking model %s %s at %s' % (problem_type, model_type, model_path))

      # Check there is a saved model.
      self.assertTrue(os.path.isfile(os.path.join(model_path, 'saved_model.pb')))

      # Must create new graphs as multiple graphs are loaded into memory.
      with tf.Graph().as_default(), tf.Session() as sess:
        meta_graph_pb = tf.saved_model.loader.load(
            sess=sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            export_dir=model_path)
        signature = meta_graph_pb.signature_def['serving_default']

        input_alias_map = {
            friendly_name: tensor_info_proto.name
            for (friendly_name, tensor_info_proto) in signature.inputs.items()}
        output_alias_map = {
            friendly_name: tensor_info_proto.name
            for (friendly_name, tensor_info_proto) in signature.outputs.items()}

        prediction_data = {
            'key': [12, 11],
            'target': [-49, -9] if problem_type == 'regression' else ['100', '101'],
            'num_id': [11, 10],
            'num_scale': [22.29, 5.20],
            'str_one_hot': ['brown', 'brown'],
            'str_embedding': ['def', 'def'],
            'str_bow': ['drone', 'drone truck bike truck'],
            'str_tfidf': ['bike train train car', 'train']}
        if with_image:
          image_bytes = []
          for image_file in [self._image_files[0], self._image_files[2]]:
            with file_io.FileIO(image_file, 'r') as ff:
              image_bytes.append(base64.urlsafe_b64encode(ff.read()))

          prediction_data.update({'image': image_bytes})

        # Convert the prediciton data to csv.
        csv_header = [col['name']
                      for col in self._schema
                      if (has_target or col['name'] != 'target')]
        if not has_target:
          del prediction_data['target']

        csv_data = []
        for i in range(2):
          data = [str(prediction_data[name][i]) for name in csv_header]
          csv_data.append(','.join(data))

        # Test the *_alias_maps have the expected keys
        expected_output_keys = ['predicted', 'key']
        if has_target:
          expected_output_keys.append('target')
        if problem_type == 'classification':
          expected_output_keys.extend(['score', 'score_2', 'score_3', 'predicted_2', 'predicted_3'])

        self.assertEqual(1, len(input_alias_map.keys()))
        self.assertItemsEqual(expected_output_keys, output_alias_map.keys())

        _, csv_tensor_name = input_alias_map.items()[0]
        result = sess.run(fetches=output_alias_map,
                          feed_dict={csv_tensor_name: csv_data})

        self.assertItemsEqual(expected_output_keys, result.keys())
        self.assertEqual([12, 11], result['key'].flatten().tolist())

  

if __name__ == '__main__':
    unittest.main()
