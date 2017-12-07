from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import cStringIO
import json
import os
from PIL import Image
import pandas as pd
import six
import shutil
import tensorflow as tf
import tempfile


from tensorflow.contrib.learn.python.learn.utils import input_fn_utils

from tensorflow.contrib import lookup
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_arg_scope
from tensorflow.python.lib.io import file_io

# ------------------------------------------------------------------------------
# public constants. Changing these could break user's code
# ------------------------------------------------------------------------------

# Individual transforms
IDENTITY_TRANSFORM = 'identity'
SCALE_TRANSFORM = 'scale'
ONE_HOT_TRANSFORM = 'one_hot'
MULTI_HOT_TRANSFORM = 'multi_hot'
TARGET_TRANSFORM = 'target'
IMAGE_TRANSFORM = 'image_to_vec'

# ------------------------------------------------------------------------------
# internal constants.
# ------------------------------------------------------------------------------

# Files
SCHEMA_FILE = 'schema.json'
FEATURES_FILE = 'features.json'
STATS_FILE = 'stats.json'
VOCAB_ANALYSIS_FILE = 'vocab_%s.csv'

# Transform collections
NUMERIC_TRANSFORMS = [IDENTITY_TRANSFORM, SCALE_TRANSFORM]
CATEGORICAL_TRANSFORMS = [ONE_HOT_TRANSFORM]
TEXT_TRANSFORMS = [MULTI_HOT_TRANSFORM]

# If the features file is missing transforms, apply these.
DEFAULT_NUMERIC_TRANSFORM = IDENTITY_TRANSFORM
DEFAULT_CATEGORICAL_TRANSFORM = ONE_HOT_TRANSFORM

# BigQuery Schema values supported
INTEGER_SCHEMA = 'integer'
FLOAT_SCHEMA = 'float'
STRING_SCHEMA = 'string'
NUMERIC_SCHEMA = [INTEGER_SCHEMA, FLOAT_SCHEMA]

# Inception Checkpoint
INCEPTION_V3_CHECKPOINT = 'gs://cloud-ml-data/img/flower_photos/inception_v3_2016_08_28.ckpt'
INCEPTION_EXCLUDED_VARIABLES = ['InceptionV3/AuxLogits', 'InceptionV3/Logits', 'global_step']

_img_buf = cStringIO.StringIO()
Image.new('RGB', (16, 16)).save(_img_buf, 'jpeg')
IMAGE_DEFAULT_STRING = base64.urlsafe_b64encode(_img_buf.getvalue())

IMAGE_BOTTLENECK_TENSOR_SIZE = 2048

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# start of transform functions
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def _scale(x, min_x_value, max_x_value, output_min, output_max):
  """Scale a column to [output_min, output_max].

  Assumes the columns's range is [min_x_value, max_x_value]. If this is not
  true at training or prediction time, the output value of this scale could be
  outside the range [output_min, output_max].

  Raises:
    ValueError: if min_x_value = max_x_value, as the column is constant.
  """

  if round(min_x_value - max_x_value, 7) == 0:
    # There is something wrong with the data.
    # Why round to 7 places? It's the same as unittest's assertAlmostEqual.
    raise ValueError('In make_scale_tito, min_x_value == max_x_value')

  def _scale(x):
    min_x_valuef = tf.to_float(min_x_value)
    max_x_valuef = tf.to_float(max_x_value)
    output_minf = tf.to_float(output_min)
    output_maxf = tf.to_float(output_max)
    return ((((tf.to_float(x) - min_x_valuef) * (output_maxf - output_minf)) /
            (max_x_valuef - min_x_valuef)) + output_minf)

  return _scale(x)


def _string_to_int(x, vocab):
  """Given a vocabulary and a string tensor `x`, maps `x` into an int tensor.
  Args:
    x: A `Column` representing a string value.
    vocab: list of strings.

  Returns:
    A `Column` where each string value is mapped to an integer representing
    its index in the vocab. Out of vocab values are mapped to len(vocab).
  """

  def _map_to_int(x):
    """Maps string tensor into indexes using vocab.

    Args:
      x : a Tensor/SparseTensor of string.
    Returns:
      a Tensor/SparseTensor of indexes (int) of the same shape as x.
    """
    table = lookup.index_table_from_tensor(
        vocab,
        default_value=len(vocab))
    return table.lookup(x)

  return _map_to_int(x)


def _make_image_to_vec_tito(feature_name, tmp_dir=None, checkpoint=None):
  """Creates a tensor-in-tensor-out function that produces embeddings from image bytes.

  Image to embedding is implemented with Tensorflow's inception v3 model and a pretrained
  checkpoint. It returns 1x2048 'PreLogits' embeddings for each image.

  Args:
    feature_name: The name of the feature. Used only to identify the image tensors so
      we can get gradients for probe in image prediction explaining.
    tmp_dir: a local directory that is used for downloading the checkpoint. If
      non, a temp folder will be made and deleted.
    checkpoint: the inception v3 checkpoint gs or local path. If None, default checkpoint
      is used.

  Returns: a tensor-in-tensor-out function that takes image string tensor and returns embeddings.
  """

  def _image_to_vec(image_str_tensor):

    def _decode_and_resize(image_tensor):
      """Decodes jpeg string, resizes it and returns a uint8 tensor."""

      # These constants are set by Inception v3's expectations.
      height = 299
      width = 299
      channels = 3

      image_tensor = tf.where(tf.equal(image_tensor, ''), IMAGE_DEFAULT_STRING, image_tensor)

      # Fork by whether image_tensor value is a file path, or a base64 encoded string.
      slash_positions = tf.equal(tf.string_split([image_tensor], delimiter="").values, '/')
      is_file_path = tf.cast(tf.count_nonzero(slash_positions), tf.bool)

      # The following two functions are required for tf.cond. Note that we can not replace them
      # with lambda. According to TF docs, if using inline lambda, both branches of condition
      # will be executed. The workaround is to use a function call.
      def _read_file():
        return tf.read_file(image_tensor)

      def _decode_base64():
        return tf.decode_base64(image_tensor)

      image = tf.cond(is_file_path, lambda: _read_file(), lambda: _decode_base64())
      image = tf.image.decode_jpeg(image, channels=channels)
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
      image = tf.squeeze(image, squeeze_dims=[0])
      image = tf.cast(image, dtype=tf.uint8)
      return image

    # The CloudML Prediction API always "feeds" the Tensorflow graph with
    # dynamic batch sizes e.g. (?,).  decode_jpeg only processes scalar
    # strings because it cannot guarantee a batch of images would have
    # the same output size.  We use tf.map_fn to give decode_jpeg a scalar
    # string from dynamic batches.
    image = tf.map_fn(_decode_and_resize, image_str_tensor, back_prop=False, dtype=tf.uint8)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # "gradients_[feature_name]" will be used for computing integrated gradients.
    image = tf.identity(image, name='gradients_' + feature_name)
    image = tf.subtract(image, 0.5)
    inception_input = tf.multiply(image, 2.0)

    # Build Inception layers, which expect a tensor of type float from [-1, 1)
    # and shape [batch_size, height, width, channels].
    with tf.contrib.slim.arg_scope(inception_v3_arg_scope()):
      _, end_points = inception_v3(inception_input, is_training=False)

    embeddings = end_points['PreLogits']
    inception_embeddings = tf.squeeze(embeddings, [1, 2], name='SpatialSqueeze')
    return inception_embeddings

  def _tito_from_checkpoint(tito_in, checkpoint, exclude):
    """ Create an all-constants tito function from an original tito function.

    Given a tensor-in-tensor-out function which contains variables and a checkpoint path,
    create a new tensor-in-tensor-out function which includes only constants, and can be
    used in tft.map.
    """

    def _tito_out(tensor_in):
      checkpoint_dir = tmp_dir
      if tmp_dir is None:
        checkpoint_dir = tempfile.mkdtemp()

      g = tf.Graph()
      with g.as_default():
        si = tf.placeholder(dtype=tensor_in.dtype, shape=tensor_in.shape, name=tensor_in.op.name)
        so = tito_in(si)
        all_vars = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
        saver = tf.train.Saver(all_vars)
        # Downloading the checkpoint from GCS to local speeds up saver.restore() a lot.
        checkpoint_tmp = os.path.join(checkpoint_dir, 'checkpoint')
        with file_io.FileIO(checkpoint, 'r') as f_in, file_io.FileIO(checkpoint_tmp, 'w') as f_out:
          f_out.write(f_in.read())
        with tf.Session() as sess:
          saver.restore(sess, checkpoint_tmp)
          output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                          g.as_graph_def(),
                                                                          [so.op.name])
        file_io.delete_file(checkpoint_tmp)
        if tmp_dir is None:
          shutil.rmtree(checkpoint_dir)

      tensors_out = tf.import_graph_def(output_graph_def,
                                        input_map={si.name: tensor_in},
                                        return_elements=[so.name])
      return tensors_out[0]

    return _tito_out

  if not checkpoint:
    checkpoint = INCEPTION_V3_CHECKPOINT
  return _tito_from_checkpoint(_image_to_vec, checkpoint, INCEPTION_EXCLUDED_VARIABLES)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# end of transform functions
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def make_preprocessing_fn(output_dir, features, keep_target):
  """Makes a preprocessing function.

  Args:
    output_dir: folder path that contains the vocab and stats files.
    features: the features dict

  Returns:
    a function that takes a dict of input tensors
  """
  def preprocessing_fn(inputs):
    """Preprocessing function.

    Args:
      inputs: dictionary of raw input tensors

    Returns:
      A dictionary of transformed tensors
    """
    stats = json.loads(
      file_io.read_file_to_string(
          os.path.join(output_dir, STATS_FILE)).decode())

    result = {}
    for name, transform in six.iteritems(features):
      transform_name = transform['transform']
      source_column = transform['source_column']

      if transform_name == TARGET_TRANSFORM:
        if not keep_target:
          continue
        if file_io.file_exists(os.path.join(output_dir, VOCAB_ANALYSIS_FILE % source_column)):
          transform_name = 'one_hot'
        else:
          transform_name = 'identity'

      if transform_name == 'identity':
        result[name] = inputs[source_column]
      elif transform_name == 'scale':
        result[name] = _scale(
            inputs[name],
            min_x_value=stats['column_stats'][source_column]['min'],
            max_x_value=stats['column_stats'][source_column]['max'],
            output_min=transform.get('value', 1) * (-1),
            output_max=transform.get('value', 1))
      elif transform_name in [ONE_HOT_TRANSFORM, MULTI_HOT_TRANSFORM]:
        vocab, ex_count = read_vocab_file(
            os.path.join(output_dir, VOCAB_ANALYSIS_FILE % source_column))
        if transform_name == MULTI_HOT_TRANSFORM:
          separator = transform.get('separator', ' ')
          tokens = tf.string_split(inputs[source_column], separator)
          result[name] = _string_to_int(tokens, vocab)
        else:
          result[name] = _string_to_int(inputs[source_column], vocab)
      elif transform_name == IMAGE_TRANSFORM:
        make_image_to_vec_fn = _make_image_to_vec_tito(
            name, checkpoint=transform.get('checkpoint', None))
        result[name] = make_image_to_vec_fn(inputs[source_column])
      else:
        raise ValueError('unknown transform %s' % transform_name)
    return result

  return preprocessing_fn


def csv_header_and_defaults(features, schema, stats, keep_target):
  """Gets csv header and default lists."""

  target_name = get_target_name(features)
  if keep_target and not target_name:
    raise ValueError('Cannot find target transform')

  csv_header = []
  record_defaults = []
  for col in schema:
    if not keep_target and col['name'] == target_name:
      continue

    # Note that numerical key columns do not have a stats entry, hence the use
    # of get(col['name'], {})
    csv_header.append(col['name'])
    if col['type'].lower() == INTEGER_SCHEMA:
      dtype = tf.int64
      default = int(stats['column_stats'].get(col['name'], {}).get('mean', 0))
    elif col['type'].lower() == FLOAT_SCHEMA:
      dtype = tf.float32
      default = float(stats['column_stats'].get(col['name'], {}).get('mean', 0.0))
    else:
      dtype = tf.string
      default = ''

    record_defaults.append(tf.constant([default], dtype=dtype))

  return csv_header, record_defaults


def build_csv_serving_tensors_for_transform_step(analysis_path,
                                                 features,
                                                 schema,
                                                 stats,
                                                 keep_target):
  """Builds a serving function starting from raw csv.

  This should only be used by transform.py (the transform step), and the

  For image columns, the image should be a base64 string encoding the image.
  The output of this function will transform that image to a 2048 long vector
  using the inception model.
  """

  csv_header, record_defaults = csv_header_and_defaults(features, schema, stats, keep_target)

  placeholder = tf.placeholder(dtype=tf.string, shape=(None,),
                               name='csv_input_placeholder')
  tensors = tf.decode_csv(placeholder, record_defaults)
  raw_features = dict(zip(csv_header, tensors))

  transform_fn = make_preprocessing_fn(analysis_path, features, keep_target)
  transformed_tensors = transform_fn(raw_features)

  transformed_features = {}
  # Expand the dims of non-sparse tensors
  for k, v in six.iteritems(transformed_tensors):
    if isinstance(v, tf.Tensor) and v.get_shape().ndims == 1:
      transformed_features[k] = tf.expand_dims(v, -1)
    else:
      transformed_features[k] = v

  return input_fn_utils.InputFnOps(
      transformed_features, None, {"csv_example": placeholder})


def get_target_name(features):
  for name, transform in six.iteritems(features):
    if transform['transform'] == TARGET_TRANSFORM:
      return name

  return None


def read_vocab_file(file_path):
  """Reads a vocab file to memeory.

  Args:
    file_path: Each line of the vocab is in the form "token,example_count"

  Returns:
    Two lists, one for the vocab, and one for just the example counts.
  """
  with file_io.FileIO(file_path, 'r') as f:
    vocab_pd = pd.read_csv(
        f,
        header=None,
        names=['vocab', 'count'],
        dtype=str,  # Prevent pd from converting numerical categories.
        na_filter=False)  # Prevent pd from converting 'NA' to a NaN.

  vocab = vocab_pd['vocab'].tolist()
  ex_count = vocab_pd['count'].astype(int).tolist()

  return vocab, ex_count


def get_transformed_feature_indices(features, stats):
  """Returns information about the transformed features.

  Returns:
    List in the from
    [(transformed_feature_name, {size: int, index_start: int})]
  """

  feature_indices = []
  index_start = 1
  for name, transform in sorted(six.iteritems(features)):
    transform_name = transform['transform']
    source_column = transform['source_column']
    info = {}
    if transform_name in [IDENTITY_TRANSFORM, SCALE_TRANSFORM]:
      info['size'] = 1
    elif transform_name in [ONE_HOT_TRANSFORM, MULTI_HOT_TRANSFORM]:
      info['size'] = stats['column_stats'][source_column]['vocab_size']
    elif transform_name == IMAGE_TRANSFORM:
      info['size'] = IMAGE_BOTTLENECK_TENSOR_SIZE
    elif transform_name == TARGET_TRANSFORM:
      info['size'] = 0
    else:
      raise ValueError('xgboost does not support transform "%s"' % transform)

    info['index_start'] = index_start
    index_start += info['size']
    feature_indices.append((name, info))

  return feature_indices


def create_feature_map(features, feature_indices, output_dir):
  """Returns feature_map about the transformed features.

  feature_map includes information such as:
    1, cat1=0
    2, cat1=1
    3, numeric1
    ...
  Returns:
    List in the from
    [(index, feature_description)]
  """
  feature_map = []
  for name, info in feature_indices:
    transform_name = features[name]['transform']
    source_column = features[name]['source_column']
    if transform_name in [IDENTITY_TRANSFORM, SCALE_TRANSFORM]:
      feature_map.append((info['index_start'], name))
    elif transform_name in [ONE_HOT_TRANSFORM, MULTI_HOT_TRANSFORM]:
      vocab, _ = read_vocab_file(
          os.path.join(output_dir, VOCAB_ANALYSIS_FILE % source_column))
      for i, word in enumerate(vocab):
        if transform_name == ONE_HOT_TRANSFORM:
          feature_map.append((info['index_start'] + i, '%s=%s' % (source_column, word)))
        elif transform_name == MULTI_HOT_TRANSFORM:
          feature_map.append((info['index_start'] + i, '%s has "%s"' % (source_column, word)))
    elif transform_name == IMAGE_TRANSFORM:
      for i in range(info['size']):
        feature_map.append((info['index_start'] + i, '%s image feature %d' % (source_column, i)))

  return feature_map
