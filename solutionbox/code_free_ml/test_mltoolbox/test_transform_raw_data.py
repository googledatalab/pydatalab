from __future__ import absolute_import
from __future__ import print_function

import json
import os
import shutil
import subprocess
import tempfile
import unittest
import uuid

import tensorflow as tf
from tensorflow.python.lib.io import file_io

import google.datalab as dl
import google.datalab.bigquery as bq

CODE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'mltoolbox', 'code_free_ml', 'data'))


class TestTransformRawData(unittest.TestCase):
  """Tests for applying a saved model"""

  def _create_test_data(self):
    """Makes local test data.

    It first creates a csv input file, then uses local analysis to produce analysis data,
    including transform_fn, schema file, transforms file.
    """
    self.working_dir = tempfile.mkdtemp()
    self.source_dir = os.path.join(self.working_dir, 'source')
    self.analysis_dir = os.path.join(self.working_dir, 'analysis')
    self.output_dir = os.path.join(self.working_dir, 'output')
    file_io.create_dir(self.source_dir)

    # Make csv input file
    self.csv_input_filepath = os.path.join(self.source_dir, 'input.csv')
    file_io.write_string_to_file(
        self.csv_input_filepath,
        '1,1,Monday,23.0,gs://cloud-ml-data/img/flower_photos/daisy/15207766_fc2f1d692c_n.jpg\n' +
        '2,0,Friday,18.0,gs://cloud-ml-data/img/flower_photos/tulips/6876631336_54bf150990.jpg\n' +
        '3,0,Sunday,12.0,gs://cloud-ml-data/img/flower_photos/roses/3705716290_cb7d803130_n.jpg\n')

    # Call analyze_data.py to create analysis results.
    schema = [{'name': 'key_col', 'type': 'INTEGER'},
              {'name': 'target_col', 'type': 'FLOAT'},
              {'name': 'cat_col', 'type': 'STRING'},
              {'name': 'num_col', 'type': 'FLOAT'},
              {'name': 'img_col', 'type': 'STRING'}]
    schema_file = os.path.join(self.source_dir, 'schema.json')
    file_io.write_string_to_file(schema_file, json.dumps(schema))
    features = {'key_col': {'transform': 'key'},
                'target_col': {'transform': 'target'},
                'cat_col': {'transform': 'one_hot'},
                'num_col': {'transform': 'identity'},
                'img_col': {'transform': 'image_to_vec'}}
    features_file = os.path.join(self.source_dir, 'features.json')
    file_io.write_string_to_file(features_file, json.dumps(features))
    cmd = ['python ' + os.path.join(CODE_PATH, 'analyze_data.py'),
           '--output-dir=' + self.analysis_dir,
           '--csv-file-pattern=' + self.csv_input_filepath,
           '--csv-schema-file=' + schema_file,
           '--features-file=' + features_file]
    subprocess.check_call(' '.join(cmd), shell=True)

  def test_local_csv_transform(self):
    """Test transfrom from local csv files."""
    try:
      self._create_test_data()

      cmd = ['python ' + os.path.join(CODE_PATH, 'transform_raw_data.py'),
             '--csv-file-pattern=' + self.csv_input_filepath,
             '--analyze-output-dir=' + self.analysis_dir,
             '--output-filename-prefix=features',
             '--output-dir=' + self.output_dir]
      subprocess.check_call(' '.join(cmd), shell=True)

      # Read the tf record file. There should only be one file.
      record_filepath = os.path.join(self.output_dir,
                                     'features-00000-of-00001.tfrecord.gz')
      options = tf.python_io.TFRecordOptions(
          compression_type=tf.python_io.TFRecordCompressionType.GZIP)
      serialized_examples = list(tf.python_io.tf_record_iterator(record_filepath, options=options))
      self.assertEqual(len(serialized_examples), 3)

      example = tf.train.Example()
      example.ParseFromString(serialized_examples[0])

      transformed_number = example.features.feature['num_col'].float_list.value[0]
      self.assertAlmostEqual(transformed_number, 23.0)
      transformed_category = example.features.feature['cat_col'].int64_list.value[0]
      self.assertEqual(transformed_category, 2)
      image_bytes = example.features.feature['img_col'].float_list.value
      self.assertEqual(len(image_bytes), 2048)
      self.assertTrue(any(x != 0.0 for x in image_bytes))

    finally:
      pass
      shutil.rmtree(self.working_dir)

  def test_local_bigquery_transform(self):
    """Test transfrom locally, but the data comes from bigquery."""
    try:
      self._create_test_data()

      # Make a BQ table, and insert 1 row.
      project_id = dl.Context.default().project_id
      dataset_name = 'test_transform_raw_data_%s' % uuid.uuid4().hex
      table_name = 'tmp_table'

      dataset = bq.Dataset((project_id, dataset_name)).create()
      table = bq.Table((project_id, dataset_name, table_name))
      table.create([{'name': 'key_col', 'type': 'INTEGER'},
                    {'name': 'target_col', 'type': 'FLOAT'},
                    {'name': 'cat_col', 'type': 'STRING'},
                    {'name': 'num_col', 'type': 'FLOAT'},
                    {'name': 'img_col', 'type': 'STRING'}])
      data = [
          {
           'key_col': 1,
           'target_col': 1.0,
           'cat_col': 'Monday',
           'num_col': 23.0,
           'img_col': 'gs://cloud-ml-data/img/flower_photos/daisy/15207766_fc2f1d692c_n.jpg',
          },
      ]
      table.insert(data=data)

      cmd = ['python ' + os.path.join(CODE_PATH, 'transform_raw_data.py'),
             '--bigquery-table=%s.%s.%s' % (project_id, dataset_name, table_name),
             '--analyze-output-dir=' + self.analysis_dir,
             '--output-filename-prefix=features',
             '--project-id=' + project_id,
             '--output-dir=' + self.output_dir]
      subprocess.check_call(' '.join(cmd), shell=True)

      # Read the tf record file. There should only be one file.
      record_filepath = os.path.join(self.output_dir,
                                     'features-00000-of-00001.tfrecord.gz')
      options = tf.python_io.TFRecordOptions(
          compression_type=tf.python_io.TFRecordCompressionType.GZIP)
      serialized_examples = list(tf.python_io.tf_record_iterator(record_filepath, options=options))
      self.assertEqual(len(serialized_examples), 1)

      example = tf.train.Example()
      example.ParseFromString(serialized_examples[0])

      transformed_number = example.features.feature['num_col'].float_list.value[0]
      self.assertAlmostEqual(transformed_number, 23.0)
      transformed_category = example.features.feature['cat_col'].int64_list.value[0]
      self.assertEqual(transformed_category, 2)
      image_bytes = example.features.feature['img_col'].float_list.value
      self.assertEqual(len(image_bytes), 2048)
      self.assertTrue(any(x != 0.0 for x in image_bytes))
    finally:
      dataset.delete(delete_contents=True)
      shutil.rmtree(self.working_dir)


if __name__ == '__main__':
    unittest.main()
