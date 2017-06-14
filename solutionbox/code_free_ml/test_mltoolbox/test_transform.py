from __future__ import absolute_import
from __future__ import print_function

import json
import os
from PIL import Image
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
    os.path.dirname(__file__), '..', 'mltoolbox', 'code_free_ml'))


class TestTransformRawData(unittest.TestCase):
  """Tests for applying a saved model"""

  @classmethod
  def setUpClass(cls):

    # Set up dirs.
    cls.working_dir = tempfile.mkdtemp()
    cls.source_dir = os.path.join(cls.working_dir, 'source')
    cls.analysis_dir = os.path.join(cls.working_dir, 'analysis')
    cls.output_dir = os.path.join(cls.working_dir, 'output')
    file_io.create_dir(cls.source_dir)

    # Make test image files.
    img1_file = os.path.join(cls.source_dir, 'img1.jpg')
    image1 = Image.new('RGBA', size=(300, 300), color=(155, 0, 0))
    image1.save(img1_file)
    img2_file = os.path.join(cls.source_dir, 'img2.jpg')
    image2 = Image.new('RGBA', size=(50, 50), color=(125, 240, 0))
    image2.save(img2_file)
    img3_file = os.path.join(cls.source_dir, 'img3.jpg')
    image3 = Image.new('RGBA', size=(800, 600), color=(33, 55, 77))
    image3.save(img3_file)

    # Make csv input file
    cls.csv_input_filepath = os.path.join(cls.source_dir, 'input.csv')
    file_io.write_string_to_file(
        cls.csv_input_filepath,
        '1,1,Monday,23.0,%s\n' % img1_file +
        '2,0,Friday,18.0,%s\n' % img2_file +
        '3,0,Sunday,12.0,%s\n' % img3_file)

    # Call analyze.py to create analysis results.
    schema = [{'name': 'key_col', 'type': 'INTEGER'},
              {'name': 'target_col', 'type': 'FLOAT'},
              {'name': 'cat_col', 'type': 'STRING'},
              {'name': 'num_col', 'type': 'FLOAT'},
              {'name': 'img_col', 'type': 'STRING'}]
    schema_file = os.path.join(cls.source_dir, 'schema.json')
    file_io.write_string_to_file(schema_file, json.dumps(schema))
    features = {'key_col': {'transform': 'key'},
                'target_col': {'transform': 'target'},
                'cat_col': {'transform': 'one_hot'},
                'num_col': {'transform': 'identity'},
                'img_col': {'transform': 'image_to_vec'}}
    features_file = os.path.join(cls.source_dir, 'features.json')
    file_io.write_string_to_file(features_file, json.dumps(features))
    cmd = ['python ' + os.path.join(CODE_PATH, 'analyze.py'),
           '--output-dir=' + cls.analysis_dir,
           '--csv-file-pattern=' + cls.csv_input_filepath,
           '--csv-schema-file=' + schema_file,
           '--features-file=' + features_file]
    subprocess.check_call(' '.join(cmd), shell=True)

    # Setup a temp GCS bucket.
    cls.bucket_root = 'gs://temp_mltoolbox_test_%s' % uuid.uuid4().hex
    subprocess.check_call('gsutil mb %s' % cls.bucket_root, shell=True)

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.working_dir)
    subprocess.check_call('gsutil -m rm -r %s' % cls.bucket_root, shell=True)

  def test_local_csv_transform(self):
    """Test transfrom from local csv files."""

    cmd = ['python ' + os.path.join(CODE_PATH, 'transform.py'),
           '--csv-file-pattern=' + self.csv_input_filepath,
           '--analysis-output-dir=' + self.analysis_dir,
           '--output-filename-prefix=features',
           '--output-dir=' + self.output_dir]
    print('cmd ', ' '.join(cmd))
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

  def test_local_bigquery_transform(self):
    """Test transfrom locally, but the data comes from bigquery."""

    # Make a BQ table, and insert 1 row.
    try:
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

      img1_file = os.path.join(self.source_dir, 'img1.jpg')
      dest_file = os.path.join(self.bucket_root, 'img1.jpg')
      subprocess.check_call('gsutil cp %s %s' % (img1_file, dest_file), shell=True)

      data = [
          {
           'key_col': 1,
           'target_col': 1.0,
           'cat_col': 'Monday',
           'num_col': 23.0,
           'img_col': dest_file,
          },
      ]
      table.insert(data=data)

      cmd = ['python ' + os.path.join(CODE_PATH, 'transform.py'),
             '--bigquery-table=%s.%s.%s' % (project_id, dataset_name, table_name),
             '--analysis-output-dir=' + self.analysis_dir,
             '--output-filename-prefix=features',
             '--project-id=' + project_id,
             '--output-dir=' + self.output_dir]
      print('cmd ', ' '.join(cmd))
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
      dataset.delete(deleete_contents=True)


if __name__ == '__main__':
    unittest.main()
