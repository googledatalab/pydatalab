from __future__ import absolute_import

import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

import tensorflow as tf
from tensorflow.python.lib.io import file_io


CODE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'mltoolbox', 'code_free_ml', 'data'))


def make_csv_data(filename, num_rows, problem_type, keep_target=True):
  """Writes csv data for preprocessing and training.

  There is one csv column for each supported transform.

  Args:
    filename: writes data to local csv file.
    num_rows: how many rows of data will be generated.
    problem_type: 'classification' or 'regression'. Changes the target value.
    keep_target: if false, the csv file will have an empty column ',,' for the
        target.
  """
  random.seed(12321)

  def _drop_out(x):
    # Make 5% of the data missing
    if random.uniform(0, 1) < 0.05:
      return ''
    return x

  with open(filename, 'w') as f:
    for i in range(num_rows):
      num_id = random.randint(0, 20)
      num_scale = random.uniform(0, 30)

      str_one_hot = random.choice(['red', 'blue', 'green', 'pink', 'yellow',
                                   'brown', 'black'])
      str_embedding = random.choice(['abc', 'def', 'ghi', 'jkl', 'mno', 'pqr'])

      def _word_fn():
        return random.choice(['car', 'truck', 'van', 'bike', 'train', 'drone'])

      str_bow = [_word_fn() for _ in range(random.randint(1, 4))]
      str_tfidf = [_word_fn() for _ in range(random.randint(1, 4))]

      color_map = {'red': 2, 'blue': 6, 'green': 4, 'pink': -5, 'yellow': -6,
                   'brown': -1, 'black': -7}
      abc_map = {'abc': -1, 'def': -1, 'ghi': 1, 'jkl': 1, 'mno': 2, 'pqr': 1}
      transport_map = {'car': 5, 'truck': 10, 'van': 15, 'bike': 20,
                       'train': -25, 'drone': -30}

      # Build some model: t id the dependent variable
      t = 0.5 + 0.5 * num_id - 2.5 * num_scale
      t += color_map[str_one_hot]
      t += abc_map[str_embedding]
      t += sum([transport_map[x] for x in str_bow])
      t += sum([transport_map[x] * 0.5 for x in str_tfidf])

      if problem_type == 'classification':
        # If you cange the weights above or add more columns, look at the new
        # distribution of t values and try to divide them into 3 buckets.
        if t < -40:
          t = 100
        elif t < 0:
          t = 101
        else:
          t = 102

      str_bow = ' '.join(str_bow)
      str_tfidf = ' '.join(str_tfidf)

      num_id = _drop_out(num_id)
      num_scale = _drop_out(num_scale)
      str_one_hot = _drop_out(str_one_hot)
      str_embedding = _drop_out(str_embedding)
      str_bow = _drop_out(str_bow)
      str_tfidf = _drop_out(str_tfidf)

      if keep_target:
          csv_line = "{key},{target},{num_id},{num_scale},{str_one_hot},{str_embedding},{str_bow},{str_tfidf}\n".format( # noqa
            key=i,
            target=t,
            num_id=num_id,
            num_scale=num_scale,
            str_one_hot=str_one_hot,
            str_embedding=str_embedding,
            str_bow=str_bow,
            str_tfidf=str_tfidf)
      else:
          csv_line = "{key},{num_id},{num_scale},{str_one_hot},{str_embedding},{str_bow},{str_tfidf}\n".format(  # noqa
            key=i,
            num_id=num_id,
            num_scale=num_scale,
            str_one_hot=str_one_hot,
            str_embedding=str_embedding,
            str_bow=str_bow,
            str_tfidf=str_tfidf)
      f.write(csv_line)


class TestTrainer(unittest.TestCase):
  """Tests training.

  Runs analysze_data and transform_raw_data on generated test data. Also loads
  the exported graphs and checks they run. No validation of the test results is
  done (i.e., the training loss is not checked).
  """
  def __init__(self, *args, **kwargs):
    super(TestTrainer, self).__init__(*args, **kwargs)

    # Allow this class to be subclassed for quick tests that only care about
    # training working, not model loss/accuracy.
    self._max_steps = 2000
    self._check_model_fit = True

    # Log everything
    self._logger = logging.getLogger('TestStructuredDataLogger')
    self._logger.setLevel(logging.DEBUG)
    if not self._logger.handlers:
      self._logger.addHandler(logging.StreamHandler(stream=sys.stdout))

  def setUp(self):
    self._test_dir = tempfile.mkdtemp()

    self._analysis_output = os.path.join(self._test_dir, 'analysis_output')
    self._transform_output = os.path.join(self._test_dir, 'transform_output')
    self._train_output = os.path.join(self._test_dir, 'train_output')

    file_io.recursive_create_dir(self._analysis_output)
    file_io.recursive_create_dir(self._transform_output)
    file_io.recursive_create_dir(self._train_output)

    self._csv_train_filename = os.path.join(self._test_dir, 'train_csv_data.csv')
    self._csv_eval_filename = os.path.join(self._test_dir, 'eval_csv_data.csv')
    self._csv_predict_filename = os.path.join(self._test_dir, 'predict_csv_data.csv')
    self._schema_filename = os.path.join(self._test_dir, 'schema_file.json')
    self._features_filename = os.path.join(self._test_dir, 'features_file.json')

  def tearDown(self):
    self._logger.debug('TestTrainer: removing test dir ' + self._test_dir)
    shutil.rmtree(self._test_dir)

  def _run_analyze(self, problem_type):
    features = {
        'num_id': {'transform': 'identity'},
        'num_scale': {'transform': 'scale', 'value': 4},
        'str_one_hot': {'transform': 'one_hot'},
        'str_embedding': {'transform': 'embedding', 'embedding_dim': 3},
        'str_bow': {'transform': 'bag_of_words'},
        'str_tfidf': {'transform': 'tfidf'},
        'target': {'transform': 'target'},
        'key': {'transform': 'key'}}

    schema = [
        {'name': 'key', 'type': 'integer'},
        {'name': 'target', 'type': 'string' if problem_type == 'classification' else 'float'},
        {'name': 'num_id', 'type': 'integer'},
        {'name': 'num_scale', 'type': 'float'},
        {'name': 'str_one_hot', 'type': 'string'},
        {'name': 'str_embedding', 'type': 'string'},
        {'name': 'str_bow', 'type': 'string'},
        {'name': 'str_tfidf', 'type': 'string'}]
    self._schema = schema

    file_io.write_string_to_file(self._schema_filename, json.dumps(schema, indent=2))
    file_io.write_string_to_file(self._features_filename, json.dumps(features, indent=2))

    make_csv_data(self._csv_train_filename, 200, problem_type, True)
    make_csv_data(self._csv_eval_filename, 100, problem_type, True)
    make_csv_data(self._csv_predict_filename, 100, problem_type, False)

    cmd = ['python %s' % os.path.join(CODE_PATH, 'analyze_data.py'),
           '--output-dir=' + self._analysis_output,
           '--csv-file-pattern=' + self._csv_train_filename,
           '--csv-schema-file=' + self._schema_filename,
           '--features-file=' + self._features_filename]

    subprocess.check_call(' '.join(cmd), shell=True)

  def _run_transform(self):
    cmd = ['python %s' % os.path.join(CODE_PATH, 'transform_raw_data.py'),
           '--csv-file-pattern=' + self._csv_train_filename,
           '--analyze-output-dir=' + self._analysis_output,
           '--output-filename-prefix=features_train',
           '--output-dir=' + self._transform_output,
           '--target',
           '--shuffle']

    self._logger.debug('Running subprocess: %s \n\n' % ' '.join(cmd))
    subprocess.check_call(' '.join(cmd), shell=True)

    cmd = ['python %s' % os.path.join(CODE_PATH, 'transform_raw_data.py'),
           '--csv-file-pattern=' + self._csv_eval_filename,
           '--analyze-output-dir=' + self._analysis_output,
           '--output-filename-prefix=features_eval',
           '--output-dir=' + self._transform_output,
           '--target']

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

  def _check_model(self, problem_type, model_type):
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

        # Build batched prediction data.
        prediction_data = {
            'key': [12, 11],
            'target': [-49, -9] if problem_type == 'regression' else ['100', '101'],
            'num_id': [11, 10],
            'num_scale': [22.29, 5.20],
            'str_one_hot': ['brown', 'brown'],
            'str_embedding': ['def', 'def'],
            'str_bow': ['drone', 'drone truck bike truck'],
            'str_tfidf': ['bike train train car', 'train']}

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

  def testClassificationLinear(self):
    self._logger.debug('\n\nTesting Classification Linear')

    problem_type = 'classification'
    model_type = 'linear'
    self._run_analyze(problem_type)
    self._run_training_raw(
        problem_type=problem_type,
        model_type=model_type,
        extra_args=['--top-n=3'])
    self._check_model(
        problem_type=problem_type,
        model_type=model_type)

  def testRegressionLinear(self):
    self._logger.debug('\n\nTesting Regression Linear')

    problem_type = 'regression'
    model_type = 'linear'
    self._run_analyze(problem_type)
    self._run_transform()
    self._run_training_transform(
        problem_type=problem_type,
        model_type=model_type)
    self._check_model(
        problem_type=problem_type,
        model_type=model_type)

  def testClassificationDNN(self):
    self._logger.debug('\n\nTesting Classification DNN')

    problem_type = 'classification'
    model_type = 'dnn'
    self._run_analyze(problem_type)
    self._run_transform()
    self._run_training_transform(
        problem_type=problem_type,
        model_type=model_type,
        extra_args=['--top-n=3',
                    '--hidden-layer-size1=10',
                    '--hidden-layer-size2=5',
                    '--hidden-layer-size3=2'])
    self._check_model(
        problem_type=problem_type,
        model_type=model_type)

  def testRegressionDNN(self):
    self._logger.debug('\n\nTesting Regression DNN')

    problem_type = 'regression'
    model_type = 'dnn'
    self._run_analyze(problem_type)
    self._run_training_raw(
        problem_type=problem_type,
        model_type=model_type,
        extra_args=['--top-n=3',
                    '--hidden-layer-size1=10',
                    '--hidden-layer-size2=5',
                    '--hidden-layer-size3=2'])
    self._check_model(
        problem_type=problem_type,
        model_type=model_type)


if __name__ == '__main__':
    unittest.main()
