from __future__ import absolute_import
from __future__ import print_function

import json
import os
import pandas as pd
from PIL import Image
from six.moves.urllib.request import urlopen
import subprocess
import tempfile
import unittest
import xgboost as xgb

from tensorflow.python.lib.io import file_io


CODE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'xgboost'))


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
    image1 = Image.new('RGB', size=(300, 300), color=(155, 0, 0))
    image1.save(img1_file)
    img2_file = os.path.join(cls.source_dir, 'img2.jpg')
    image2 = Image.new('RGB', size=(50, 50), color=(125, 240, 0))
    image2.save(img2_file)
    img3_file = os.path.join(cls.source_dir, 'img3.jpg')
    image3 = Image.new('RGB', size=(800, 600), color=(33, 55, 77))
    image3.save(img3_file)

    # Download inception checkpoint. Note that gs url doesn't work because
    # we may not have gcloud signed in when running the test.
    url = ('https://storage.googleapis.com/cloud-ml-data/img/' +
           'flower_photos/inception_v3_2016_08_28.ckpt')
    checkpoint_path = os.path.join(cls.working_dir, "checkpoint")
    response = urlopen(url)
    with open(checkpoint_path, 'wb') as f:
      f.write(response.read())

    # Make csv input file
    cls.csv_input_filepath = os.path.join(cls.source_dir, 'input.csv')
    file_io.write_string_to_file(
        cls.csv_input_filepath,
        '1,Monday,23.0,red blue,%s\n' % img1_file +
        '0,Friday,18.0,green,%s\n' % img2_file +
        '0,Sunday,12.0,green red blue green,%s\n' % img3_file)

    # Call analyze.py to create analysis results.
    schema = [{'name': 'target_col', 'type': 'FLOAT'},
              {'name': 'cat_col', 'type': 'STRING'},
              {'name': 'num_col', 'type': 'FLOAT'},
              {'name': 'text_col', 'type': 'STRING'},
              {'name': 'img_col', 'type': 'STRING'}]
    schema_file = os.path.join(cls.source_dir, 'schema.json')
    file_io.write_string_to_file(schema_file, json.dumps(schema))
    features = {'target_col': {'transform': 'target'},
                'cat_col': {'transform': 'one_hot'},
                'num_col': {'transform': 'identity'},
                'text_col': {'transform': 'multi_hot'},
                'img_col': {'transform': 'image_to_vec', 'checkpoint': checkpoint_path}}
    features_file = os.path.join(cls.source_dir, 'features.json')
    file_io.write_string_to_file(features_file, json.dumps(features))
    cmd = ['python ' + os.path.join(CODE_PATH, 'analyze.py'),
           '--output=' + cls.analysis_dir,
           '--csv=' + cls.csv_input_filepath,
           '--schema=' + schema_file,
           '--features=' + features_file]
    subprocess.check_call(' '.join(cmd), shell=True)

  @classmethod
  def tearDownClass(cls):
    pass
    # shutil.rmtree(cls.working_dir)

  def test_local_csv_transform(self):
    """Test transfrom from local csv files."""

    cmd = ['python ' + os.path.join(CODE_PATH, 'transform.py'),
           '--csv=' + self.csv_input_filepath,
           '--analysis=' + self.analysis_dir,
           '--prefix=features',
           '--output=' + self.output_dir]
    print('cmd ', ' '.join(cmd))
    subprocess.check_call(' '.join(cmd), shell=True)

    # Verify transformed file.
    libsvm_filepath = os.path.join(self.output_dir, 'features-00000-of-00001.libsvm')
    dtrain = xgb.DMatrix(libsvm_filepath)
    self.assertTrue(2056, dtrain.num_col())
    self.assertTrue(3, dtrain.num_row())

    # Verify featuremap file.
    featuremap_filepath = os.path.join(self.output_dir, 'featuremap-00000-of-00001.txt')
    df = pd.read_csv(featuremap_filepath, names=['index', 'description'])
    pd.util.testing.assert_series_equal(pd.Series(range(1, 2056), name='index'), df['index'])
    expected_descriptions = ['cat_col=Sunday', 'cat_col=Monday', 'img_col image feature 1000',
                             'num_col', 'text_col has "blue"']
    self.assertTrue(all(x in df['description'].values for x in expected_descriptions))


if __name__ == '__main__':
    unittest.main()
