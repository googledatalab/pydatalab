from __future__ import absolute_import
from __future__ import print_function

import base64
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import uuid
import apache_beam as beam
from PIL import Image
import six

import tensorflow as tf
from tensorflow.python.lib.io import file_io

import tensorflow_transform as tft
from tensorflow_transform.beam import impl as tft_impl
from tensorflow_transform.beam import tft_beam_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io

import google.datalab as dl
import google.datalab.bigquery as bq

CODE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'mltoolbox', 'code_free_ml', 'data'))


class TestTransformRawData(unittest.TestCase):
  """Tests for applying a saved model"""

  def _create_test_data(self):
    """Makes local test data.

    The fllowing files and folders will be created in self.output_folder:

    self.output_folder/
        features.json
        img.png
        input.csv
        schema.json
        raw_metadata/
            (tft metadata files)
        transformed_metadata/
            (tft metadata files)
        transform_fn/
            (tft saved model file)
    """
    self.output_folder = tempfile.mkdtemp()

    # Make image file
    self.img_filepath = os.path.join(self.output_folder, 'img.png')
    image = Image.new('RGBA', size=(50, 50), color=(155, 0, 0))
    image.save(self.img_filepath, 'png')

    # Make csv input file
    self.csv_input_filepath = os.path.join(self.output_folder, 'input.csv')
    file_io.write_string_to_file(
        self.csv_input_filepath,
        '23.0,%s' % self.img_filepath)

    # Make schema file
    self.schema_filepath = os.path.join(self.output_folder, 'schema.json')
    file_io.write_string_to_file(
        self.schema_filepath,
        json.dumps([{'name': 'num_col', 'type': 'FLOAT'},
                    {'name': 'img_col', 'type': 'STRING'}]))

    # Make features file
    self.features_filepath = os.path.join(self.output_folder, 'features.json')
    file_io.write_string_to_file(
        self.features_filepath,
        json.dumps({'num_col': {'transform': 'target'},
                    'img_col': {'transform': 'img_url_to_vec'}}))

    # Run a local beam job to make the transform_fn
    with beam.Pipeline('DirectRunner'):
      with tft_impl.Context(temp_dir=os.path.join(self.output_folder, 'tmp')):
        def preprocessing_fn(inputs):
          return {'img_col': tft.map(tf.decode_base64, inputs['img_col']),
                  'num_col': tft.map(lambda x: tf.add(x, 1), inputs['num_col'])}

        input_data = [{'img_col': base64.urlsafe_b64encode('abcd'), 'num_col': 3}]

        input_metadata = dataset_metadata.DatasetMetadata(
            schema=dataset_schema.from_feature_spec(
                {'img_col': tf.FixedLenFeature(shape=[], dtype=tf.string),
                 'num_col': tf.FixedLenFeature(shape=[], dtype=tf.float32)}))

        (dataset, train_metadata), transform_fn = (
            (input_data, input_metadata)
            | 'AnalyzeAndTransform'  # noqa: W503
            >> tft_impl.AnalyzeAndTransformDataset(preprocessing_fn))  # noqa: W503

        # WriteTransformFn writes transform_fn and metadata
        _ = (transform_fn  # noqa: F841
             | 'WriteTransformFn'  # noqa: W503
             >> tft_beam_io.WriteTransformFn(self.output_folder))  # noqa: W503

        metadata_io.write_metadata(
            metadata=input_metadata,
            path=os.path.join(self.output_folder, 'raw_metadata'))

  def test_local_csv_transform(self):
    """Test transfrom from local csv files."""
    try:
      self._create_test_data()
      tfex_dir = os.path.join(self.output_folder, 'test_results')
      cmd = ['python ' + os.path.join(CODE_PATH, 'transform_raw_data.py'),
             '--csv-file-pattern=' + self.csv_input_filepath,
             '--analyze-output-dir=' + self.output_folder,
             '--output-filename-prefix=features',
             '--output-dir=' + tfex_dir]
      subprocess.check_call(' '.join(cmd), shell=True)

      # Read the tf record file. There should only be one file.
      record_filepath = os.path.join(tfex_dir,
                                     'features-00000-of-00001.tfrecord.gz')
      options = tf.python_io.TFRecordOptions(
          compression_type=tf.python_io.TFRecordCompressionType.GZIP)
      serialized_example = next(
          tf.python_io.tf_record_iterator(
              record_filepath,
              options=options))
      example = tf.train.Example()
      example.ParseFromString(serialized_example)

      transformed_number = example.features.feature['num_col'].float_list.value[0]
      self.assertAlmostEqual(transformed_number, 24.0)

      image_bytes = example.features.feature['img_col'].bytes_list.value[0]
      raw_img = Image.open(self.img_filepath).convert('RGB')
      img_file = six.BytesIO()
      raw_img.save(img_file, 'jpeg')
      expected_image_bytes = img_file.getvalue()

      self.assertEqual(image_bytes, expected_image_bytes)

    finally:
      shutil.rmtree(self.output_folder)

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
      table.create([{'name': 'num_col', 'type': 'FLOAT'},
                    {'name': 'img_col', 'type': 'STRING'}])
      table.insert(data=[{'num_col': 23.0, 'img_col': self.img_filepath}])

      tfex_dir = os.path.join(self.output_folder, 'test_results')
      cmd = ['python ' + os.path.join(CODE_PATH, 'transform_raw_data.py'),
             '--bigquery-table=%s.%s.%s' % (project_id, dataset_name, table_name),
             '--analyze-output-dir=' + self.output_folder,
             '--output-filename-prefix=features',
             '--project-id=' + project_id,
             '--output-dir=' + tfex_dir]
      subprocess.check_call(' '.join(cmd), shell=True)

      # Read the tf record file. There should only be one file.
      record_filepath = os.path.join(tfex_dir,
                                     'features-00000-of-00001.tfrecord.gz')
      options = tf.python_io.TFRecordOptions(
          compression_type=tf.python_io.TFRecordCompressionType.GZIP)
      serialized_example = next(
          tf.python_io.tf_record_iterator(
              record_filepath,
              options=options))
      example = tf.train.Example()
      example.ParseFromString(serialized_example)

      transformed_number = example.features.feature['num_col'].float_list.value[0]
      self.assertAlmostEqual(transformed_number, 24.0)

      image_bytes = example.features.feature['img_col'].bytes_list.value[0]
      raw_img = Image.open(self.img_filepath).convert('RGB')
      img_file = six.BytesIO()
      raw_img.save(img_file, 'jpeg')
      expected_image_bytes = img_file.getvalue()

      self.assertEqual(image_bytes, expected_image_bytes)
    finally:
      dataset.delete(delete_contents=True)
      shutil.rmtree(self.output_folder)


if __name__ == '__main__':
    unittest.main()
