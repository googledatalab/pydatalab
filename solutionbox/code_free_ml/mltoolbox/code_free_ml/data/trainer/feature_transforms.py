from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import collections
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
EMBEDDING_TRANSFROM = 'embedding'
BOW_TRANSFORM = 'bag_of_words'
TFIDF_TRANSFORM = 'tfidf'
KEY_TRANSFORM = 'key'
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
CATEGORICAL_TRANSFORMS = [ONE_HOT_TRANSFORM, EMBEDDING_TRANSFROM]
TEXT_TRANSFORMS = [BOW_TRANSFORM, TFIDF_TRANSFORM]

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
    table = lookup.string_to_index_table_from_tensor(
        vocab,
        default_value=len(vocab))
    return table.lookup(x)

  return _map_to_int(x)


# TODO(brandondura): update this to not depend on tf layer's feature column
# 'sum' combiner in the future.
def _tfidf(x, reduced_term_freq, vocab_size, corpus_size):
  """Maps the terms in x to their (1/doc_length) * inverse document frequency.
  Args:
    x: A `Column` representing int64 values (most likely that are the result
        of calling string_to_int on a tokenized string).
    reduced_term_freq: A dense tensor of shape (vocab_size,) that represents
        the count of the number of documents with each term. So vocab token i (
        which is an int) occures in reduced_term_freq[i] examples in the corpus.
        This means reduced_term_freq should have a count for out-of-vocab tokens
    vocab_size: An int - the count of vocab used to turn the string into int64s
        including any out-of-vocab ids
    corpus_size: A scalar count of the number of documents in the corpus
  Returns:
    A `Column` where each int value is mapped to a double equal to
    (1 if that term appears in that row, 0 otherwise / the number of terms in
    that row) * the log of (the number of rows in `x` / (1 + the number of
    rows in `x` where the term appears at least once))
  NOTE:
    This is intented to be used with the feature_column 'sum' combiner to arrive
    at the true term frequncies.
  """

  def _map_to_vocab_range(x):
    """Enforces that the vocab_ids in x are positive."""
    return tf.SparseTensor(
        indices=x.indices,
        values=tf.mod(x.values, vocab_size),
        dense_shape=x.dense_shape)

  def _map_to_tfidf(x):
    """Calculates the inverse document frequency of terms in the corpus.
    Args:
      x : a SparseTensor of int64 representing string indices in vocab.
    Returns:
      The tf*idf values
    """
    # Add one to the reduced term freqnencies to avoid dividing by zero.
    idf = tf.log(tf.to_double(corpus_size) / (
        1.0 + tf.to_double(reduced_term_freq)))

    dense_doc_sizes = tf.to_double(tf.sparse_reduce_sum(tf.SparseTensor(
        indices=x.indices,
        values=tf.ones_like(x.values),
        dense_shape=x.dense_shape), 1))

    # For every term in x, divide the idf by the doc size.
    # The two gathers both result in shape <sum_doc_sizes>
    idf_over_doc_size = (tf.gather(idf, x.values) /
                         tf.gather(dense_doc_sizes, x.indices[:, 0]))

    return tf.SparseTensor(
        indices=x.indices,
        values=idf_over_doc_size,
        dense_shape=x.dense_shape)

  cleaned_input = _map_to_vocab_range(x)

  weights = _map_to_tfidf(cleaned_input)
  return tf.to_float(weights)


# TODO(brandondura): update this to not depend on tf layer's feature column
# 'sum' combiner in the future.
def _bag_of_words(x):
  """Computes bag of words weights

  Note the return type is a float sparse tensor, not a int sparse tensor. This
  is so that the output types batch tfidf, and any downstream transformation
  in tf layers during training can be applied to both.
  """
  def _bow(x):
    """Comptue BOW weights.

    As tf layer's sum combiner is used, the weights can be just ones. Tokens are
    not summed together here.
    """
    return tf.SparseTensor(
      indices=x.indices,
      values=tf.to_float(tf.ones_like(x.values)),
      dense_shape=x.dense_shape)

  return _bow(x)


def _make_image_to_vec_tito(tmp_dir=None):
  """Creates a tensor-in-tensor-out function that produces embeddings from image bytes.

  Image to embedding is implemented with Tensorflow's inception v3 model and a pretrained
  checkpoint. It returns 1x2048 'PreLogits' embeddings for each image.

  Args:
    tmp_dir: a local directory that is used for downloading the checkpoint. If
      non, a temp folder will be made and deleted.

  Returns: a tensor-in-tensor-out function that takes image string tensor and returns embeddings.
  """

  def _image_to_vec(image_str_tensor):

    def _decode_and_resize(image_str_tensor):
      """Decodes jpeg string, resizes it and returns a uint8 tensor."""

      # These constants are set by Inception v3's expectations.
      height = 299
      width = 299
      channels = 3

      image = tf.where(tf.equal(image_str_tensor, ''), IMAGE_DEFAULT_STRING, image_str_tensor)
      image = tf.decode_base64(image)
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

  return _tito_from_checkpoint(_image_to_vec,
                               INCEPTION_V3_CHECKPOINT,
                               INCEPTION_EXCLUDED_VARIABLES)
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

      if transform_name == KEY_TRANSFORM:
        transform_name = 'identity'
      elif transform_name == TARGET_TRANSFORM:
        if not keep_target:
          continue
        if file_io.file_exists(os.path.join(output_dir, VOCAB_ANALYSIS_FILE % name)):
          transform_name = 'one_hot'
        else:
          transform_name = 'identity'

      if transform_name == 'identity':
        result[name] = inputs[name]
      elif transform_name == 'scale':
        result[name] = _scale(
            inputs[name],
            min_x_value=stats['column_stats'][name]['min'],
            max_x_value=stats['column_stats'][name]['max'],
            output_min=transform.get('value', 1) * (-1),
            output_max=transform.get('value', 1))
      elif transform_name in [ONE_HOT_TRANSFORM, EMBEDDING_TRANSFROM,
                              TFIDF_TRANSFORM, BOW_TRANSFORM]:
        vocab, ex_count = read_vocab_file(
            os.path.join(output_dir, VOCAB_ANALYSIS_FILE % name))

        if transform_name == TFIDF_TRANSFORM:
          tokens = tf.string_split(inputs[name], ' ')
          ids = _string_to_int(tokens, vocab)
          weights = _tfidf(
              x=ids,
              reduced_term_freq=ex_count + [0],
              vocab_size=len(vocab) + 1,
              corpus_size=stats['num_examples'])

          result[name + '_ids'] = ids
          result[name + '_weights'] = weights
        elif transform_name == BOW_TRANSFORM:
          tokens = tf.string_split(inputs[name], ' ')
          ids = _string_to_int(tokens, vocab)
          weights = _bag_of_words(x=ids)

          result[name + '_ids'] = ids
          result[name + '_weights'] = weights
        else:
          # ONE_HOT_TRANSFORM: making a dense vector is done at training
          # EMBEDDING_TRANSFROM: embedding vectors have to be done at training
          result[name] = _string_to_int(inputs[name], vocab)
      elif transform_name == IMAGE_TRANSFORM:
        make_image_to_vec_fn = _make_image_to_vec_tito()
        result[name] = make_image_to_vec_fn(inputs[name])
      else:
        raise ValueError('unknown transform %s' % transform_name)
    return result

  return preprocessing_fn


def get_transfrormed_feature_info(features, schema):
  """Returns information about the transformed features.

  Returns:
    Dict in the from
    {transformed_feature_name: {dtype: tf type, size: int or None}}. If the size
    is None, then the tensor is a sparse tensor.
  """

  info = collections.defaultdict(dict)

  for name, transform in six.iteritems(features):
    transform_name = transform['transform']

    if transform_name == IDENTITY_TRANSFORM:
      schema_type = next(col['type'].lower() for col in schema if col['name'] == name)
      if schema_type == FLOAT_SCHEMA:
        info[name]['dtype'] = tf.float32
      elif schema_type == INTEGER_SCHEMA:
        info[name]['dtype'] = tf.int64
      else:
        raise ValueError('itentity should only be applied to integer or float'
                         'columns, but was used on %s' % name)
      info[name]['size'] = 1
    elif transform_name == SCALE_TRANSFORM:
      info[name]['dtype'] = tf.float32
      info[name]['size'] = 1
    elif transform_name == ONE_HOT_TRANSFORM:
      info[name]['dtype'] = tf.int64
      info[name]['size'] = 1
    elif transform_name == EMBEDDING_TRANSFROM:
      info[name]['dtype'] = tf.int64
      info[name]['size'] = 1
    elif transform_name == BOW_TRANSFORM or transform_name == TFIDF_TRANSFORM:
      info[name + '_ids']['dtype'] = tf.int64
      info[name + '_weights']['dtype'] = tf.float32
      info[name + '_ids']['size'] = None
      info[name + '_weights']['size'] = None
    elif transform_name == KEY_TRANSFORM:
      schema_type = next(col['type'].lower() for col in schema if col['name'] == name)
      if schema_type == FLOAT_SCHEMA:
        info[name]['dtype'] = tf.float32
      elif schema_type == INTEGER_SCHEMA:
        info[name]['dtype'] = tf.int64
      else:
        info[name]['dtype'] = tf.string
      info[name]['size'] = 1
    elif transform_name == TARGET_TRANSFORM:
      # If the input is a string, it gets converted to an int (id)
      schema_type = next(col['type'].lower() for col in schema if col['name'] == name)
      if schema_type in NUMERIC_SCHEMA:
        info[name]['dtype'] = tf.float32
      else:
        info[name]['dtype'] = tf.int64
      info[name]['size'] = 1
    elif transform_name == IMAGE_TRANSFORM:
      info[name]['dtype'] = tf.float32
      info[name]['size'] = IMAGE_BOTTLENECK_TENSOR_SIZE
    else:
      raise ValueError('Unknown transfrom %s' % transform_name)

  return info


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


def build_csv_serving_tensors(analysis_path, features, schema, stats, keep_target):
  """Returns a placeholder tensor and transformed tensors."""

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


def build_csv_transforming_training_input_fn(schema,
                                             features,
                                             stats,
                                             analysis_output_dir,
                                             raw_data_file_pattern,
                                             training_batch_size,
                                             num_epochs=None,
                                             randomize_input=False,
                                             min_after_dequeue=1,
                                             reader_num_threads=1):
  """Creates training input_fn that reads raw csv data and applies transforms.

  Args:
    schema: schema list
    features: features dict
    stats: stats dict
    analysis_output_dir: output folder from analysis
    raw_data_file_pattern: file path, or list of files
    training_batch_size: An int specifying the batch size to use.
    num_epochs: numer of epochs to read from the files. Use None to read forever.
    randomize_input: If true, the input rows are read out of order. This
        randomness is limited by the min_after_dequeue value.
    min_after_dequeue: Minimum number elements in the reading queue after a
        dequeue, used to ensure a level of mixing of elements. Only used if
        randomize_input is True.
    reader_num_threads: The number of threads enqueuing data.

  Returns:
    An input_fn suitable for training that reads raw csv training data and
    applies transforms.

  """

  def raw_training_input_fn():
    """Training input function that reads raw data and applies transforms."""

    if isinstance(raw_data_file_pattern, six.string_types):
      filepath_list = [raw_data_file_pattern]
    else:
      filepath_list = raw_data_file_pattern

    files = []
    for path in filepath_list:
      files.extend(file_io.get_matching_files(path))

    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=randomize_input)

    csv_id, csv_lines = tf.TextLineReader().read_up_to(filename_queue, training_batch_size)

    queue_capacity = (reader_num_threads + 3) * training_batch_size + min_after_dequeue
    if randomize_input:
      _, batch_csv_lines = tf.train.shuffle_batch(
          tensors=[csv_id, csv_lines],
          batch_size=training_batch_size,
          capacity=queue_capacity,
          min_after_dequeue=min_after_dequeue,
          enqueue_many=True,
          num_threads=reader_num_threads)

    else:
      _, batch_csv_lines = tf.train.batch(
          tensors=[csv_id, csv_lines],
          batch_size=training_batch_size,
          capacity=queue_capacity,
          enqueue_many=True,
          num_threads=reader_num_threads)

    csv_header, record_defaults = csv_header_and_defaults(features, schema, stats, keep_target=True)
    parsed_tensors = tf.decode_csv(batch_csv_lines, record_defaults, name='csv_to_tensors')
    raw_features = dict(zip(csv_header, parsed_tensors))

    transform_fn = make_preprocessing_fn(analysis_output_dir, features, keep_target=True)
    transformed_tensors = transform_fn(raw_features)

    # Expand the dims of non-sparse tensors. This is needed by tf.learn.
    transformed_features = {}
    for k, v in six.iteritems(transformed_tensors):
      if isinstance(v, tf.Tensor) and v.get_shape().ndims == 1:
        transformed_features[k] = tf.expand_dims(v, -1)
      else:
        transformed_features[k] = v

    # Remove the target tensor, and return it directly
    target_name = get_target_name(features)
    if not target_name or target_name not in transformed_features:
      raise ValueError('Cannot find target transform in features')

    transformed_target = transformed_features.pop(target_name)

    return transformed_features, transformed_target

  return raw_training_input_fn


def build_tfexample_transfored_training_input_fn(schema,
                                                 features,
                                                 analysis_output_dir,
                                                 raw_data_file_pattern,
                                                 training_batch_size,
                                                 num_epochs=None,
                                                 randomize_input=False,
                                                 min_after_dequeue=1,
                                                 reader_num_threads=1):
  """Creates training input_fn that reads transformed tf.example files.

  Args:
    schema: schema list
    features: features dict
    analysis_output_dir: output folder from analysis
    raw_data_file_pattern: file path, or list of files
    training_batch_size: An int specifying the batch size to use.
    num_epochs: numer of epochs to read from the files. Use None to read forever.
    randomize_input: If true, the input rows are read out of order. This
        randomness is limited by the min_after_dequeue value.
    min_after_dequeue: Minimum number elements in the reading queue after a
        dequeue, used to ensure a level of mixing of elements. Only used if
        randomize_input is True.
    reader_num_threads: The number of threads enqueuing data.

  Returns:
    An input_fn suitable for training that reads transformed data in tf record
      files of tf.example.
  """

  def transformed_training_input_fn():
    """Training input function that reads transformed data."""

    if isinstance(raw_data_file_pattern, six.string_types):
      filepath_list = [raw_data_file_pattern]
    else:
      filepath_list = raw_data_file_pattern

    files = []
    for path in filepath_list:
      files.extend(file_io.get_matching_files(path))

    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=randomize_input)

    options = tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    ex_id, ex_str = tf.TFRecordReader(options=options).read_up_to(
        filename_queue, training_batch_size)

    queue_capacity = (reader_num_threads + 3) * training_batch_size + min_after_dequeue
    if randomize_input:
      _, batch_ex_str = tf.train.shuffle_batch(
          tensors=[ex_id, ex_str],
          batch_size=training_batch_size,
          capacity=queue_capacity,
          min_after_dequeue=min_after_dequeue,
          enqueue_many=True,
          num_threads=reader_num_threads)

    else:
      _, batch_ex_str = tf.train.batch(
          tensors=[ex_id, ex_str],
          batch_size=training_batch_size,
          capacity=queue_capacity,
          enqueue_many=True,
          num_threads=reader_num_threads)

    feature_spec = {}
    feature_info = get_transfrormed_feature_info(features, schema)
    for name, info in six.iteritems(feature_info):
      if info['size'] is None:
        feature_spec[name] = tf.VarLenFeature(dtype=info['dtype'])
      else:
        feature_spec[name] = tf.FixedLenFeature(shape=[info['size']], dtype=info['dtype'])

    parsed_tensors = tf.parse_example(batch_ex_str, feature_spec)

    # Expand the dims of non-sparse tensors. This is needed by tf.learn.
    transformed_features = {}
    for k, v in six.iteritems(parsed_tensors):
      if isinstance(v, tf.Tensor) and v.get_shape().ndims == 1:
        transformed_features[k] = tf.expand_dims(v, -1)
      else:
        # Sparse tensor
        transformed_features[k] = v

    # Remove the target tensor, and return it directly
    target_name = get_target_name(features)
    if not target_name or target_name not in transformed_features:
      raise ValueError('Cannot find target transform in features')

    transformed_target = transformed_features.pop(target_name)

    return transformed_features, transformed_target

  return transformed_training_input_fn


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
