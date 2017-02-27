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


"""Inception model tensorflow implementation.
"""

from enum import Enum
import json
import logging
import os
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants

from . import _inceptionlib
from . import _util


slim = tf.contrib.slim

LOGITS_TENSOR_NAME = 'logits_tensor'
IMAGE_URI_COLUMN = 'image_uri'
LABEL_COLUMN = 'label'
EMBEDDING_COLUMN = 'embedding'

BOTTLENECK_TENSOR_SIZE = 2048


class GraphMod(Enum):
  TRAIN = 1
  EVALUATE = 2
  PREDICT = 3


class GraphReferences(object):
  """Holder of base tensors used for training model using common task."""

  def __init__(self):
    self.examples = None
    self.train = None
    self.global_step = None
    self.metric_updates = []
    self.metric_values = []
    self.keys = None
    self.predictions = []


class Model(object):
  """TensorFlow model for the flowers problem."""

  def __init__(self, labels, dropout, inception_checkpoint_file):
    self.labels = labels
    self.labels.sort()
    self.dropout = dropout
    self.inception_checkpoint_file = inception_checkpoint_file

  def add_final_training_ops(self,
                             embeddings,
                             all_labels_count,
                             bottleneck_tensor_size,
                             hidden_layer_size=BOTTLENECK_TENSOR_SIZE / 4,
                             dropout_keep_prob=None):
    """Adds a new softmax and fully-connected layer for training.

     The set up for the softmax and fully-connected layers is based on:
     https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

     This function can be customized to add arbitrary layers for
     application-specific requirements.
    Args:
      embeddings: The embedding (bottleneck) tensor.
      all_labels_count: The number of all labels including the default label.
      bottleneck_tensor_size: The number of embeddings.
      hidden_layer_size: The size of the hidden_layer. Roughtly, 1/4 of the
                         bottleneck tensor size.
      dropout_keep_prob: the percentage of activation values that are retained.
    Returns:
      softmax: The softmax or tensor. It stores the final scores.
      logits: The logits tensor.
    """
    with tf.name_scope('input'):
      bottleneck_input = tf.placeholder_with_default(
          embeddings,
          shape=[None, bottleneck_tensor_size],
          name='ReshapeSqueezed')
      bottleneck_with_no_gradient = tf.stop_gradient(bottleneck_input)

      with tf.name_scope('Wx_plus_b'):
        hidden = layers.fully_connected(bottleneck_with_no_gradient,
                                        hidden_layer_size)
        # We need a dropout when the size of the dataset is rather small.
        if dropout_keep_prob:
          hidden = tf.nn.dropout(hidden, dropout_keep_prob)
        logits = layers.fully_connected(
            hidden, all_labels_count, activation_fn=None)

    softmax = tf.nn.softmax(logits, name='softmax')
    return softmax, logits

  def build_inception_graph(self):
    """Builds an inception graph and add the necessary input & output tensors.

      To use other Inception models modify this file. Also preprocessing must be
      modified accordingly.

      See tensorflow/contrib/slim/python/slim/nets/inception_v3.py for
      details about InceptionV3.

    Returns:
      input_jpeg: A placeholder for jpeg string batch that allows feeding the
                  Inception layer with image bytes for prediction.
      inception_embeddings: The embeddings tensor.
    """
    image_str_tensor = tf.placeholder(tf.string, shape=[None])

    # The CloudML Prediction API always "feeds" the Tensorflow graph with
    # dynamic batch sizes e.g. (?,).  decode_jpeg only processes scalar
    # strings because it cannot guarantee a batch of images would have
    # the same output size.  We use tf.map_fn to give decode_jpeg a scalar
    # string from dynamic batches.
    image = tf.map_fn(
        _util.decode_and_resize, image_str_tensor, back_prop=False, dtype=tf.uint8)
    # convert_image_dtype, also scales [0, uint8_max] -> [0 ,1).
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Then shift images to [-1, 1) for Inception.
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    # Build Inception layers, which expect A tensor of type float from [-1, 1)
    # and shape [batch_size, height, width, channels].
    with slim.arg_scope(_inceptionlib.inception_v3_arg_scope()):
      _, end_points = _inceptionlib.inception_v3(image, is_training=False)

    inception_embeddings = end_points['PreLogits']
    inception_embeddings = tf.squeeze(
        inception_embeddings, [1, 2], name='SpatialSqueeze')
    return image_str_tensor, inception_embeddings

  def build_graph(self, data_paths, batch_size, graph_mod):
    """Builds generic graph for training or eval."""
    tensors = GraphReferences()
    is_training = graph_mod == GraphMod.TRAIN
    if data_paths:
      _, tensors.examples = _util.read_examples(
          data_paths,
          batch_size,
          shuffle=is_training,
          num_epochs=None if is_training else 2)
    else:
      tensors.examples = tf.placeholder(tf.string, name='input', shape=(None,))

    if graph_mod == GraphMod.PREDICT:
      inception_input, inception_embeddings = self.build_inception_graph()
      # Build the Inception graph. We later add final training layers
      # to this graph. This is currently used only for prediction.
      # For training, we use pre-processed data, so it is not needed.
      embeddings = inception_embeddings
      tensors.input_jpeg = inception_input
    else:
      # For training and evaluation we assume data is preprocessed, so the
      # inputs are tf-examples.
      # Generate placeholders for examples.
      with tf.name_scope('inputs'):
        feature_map = {
            'image_uri':
                tf.FixedLenFeature(
                    shape=[], dtype=tf.string, default_value=['']),
            # Some images may have no labels. For those, we assume a default
            # label. So the number of labels is label_count+1 for the default
            # label.
            'label':
                tf.FixedLenFeature(
                    shape=[1], dtype=tf.int64,
                    default_value=[len(self.labels)]),
            'embedding':
                tf.FixedLenFeature(
                    shape=[BOTTLENECK_TENSOR_SIZE], dtype=tf.float32)
        }
        parsed = tf.parse_example(tensors.examples, features=feature_map)
        labels = tf.squeeze(parsed['label'])
        uris = tf.squeeze(parsed['image_uri'])
        embeddings = parsed['embedding']

    # We assume a default label, so the total number of labels is equal to
    # label_count+1.
    all_labels_count = len(self.labels) + 1
    with tf.name_scope('final_ops'):
      softmax, logits = self.add_final_training_ops(
          embeddings,
          all_labels_count,
          BOTTLENECK_TENSOR_SIZE,
          dropout_keep_prob=self.dropout if is_training else None)

    # Prediction is the index of the label with the highest score. We are
    # interested only in the top score.
    prediction = tf.argmax(softmax, 1)
    tensors.predictions = [prediction, softmax, embeddings]

    if graph_mod == GraphMod.PREDICT:
      return tensors

    with tf.name_scope('evaluate'):
      loss_value = loss(logits, labels)

    # Add to the Graph the Ops that calculate and apply gradients.
    if is_training:
      tensors.train, tensors.global_step = training(loss_value)
    else:
      tensors.global_step = tf.Variable(0, name='global_step', trainable=False)
      tensors.uris = uris

    # Add means across all batches.
    loss_updates, loss_op = _util.loss(loss_value)
    accuracy_updates, accuracy_op = _util.accuracy(logits, labels)

    if not is_training:
      tf.summary.scalar('accuracy', accuracy_op)
      tf.summary.scalar('loss', loss_op)

    tensors.metric_updates = loss_updates + accuracy_updates
    tensors.metric_values = [loss_op, accuracy_op]
    return tensors
    

  def build_train_graph(self, data_paths, batch_size):
    return self.build_graph(data_paths, batch_size, GraphMod.TRAIN)

  def build_eval_graph(self, data_paths, batch_size):
    return self.build_graph(data_paths, batch_size, GraphMod.EVALUATE)

  def restore_from_checkpoint(self, session, inception_checkpoint_file,
                              trained_checkpoint_file):
    """To restore model variables from the checkpoint file.

       The graph is assumed to consist of an inception model and other
       layers including a softmax and a fully connected layer. The former is
       pre-trained and the latter is trained using the pre-processed data. So
       we restore this from two checkpoint files.
    Args:
      session: The session to be used for restoring from checkpoint.
      inception_checkpoint_file: Path to the checkpoint file for the Inception
                                 graph.
      trained_checkpoint_file: path to the trained checkpoint for the other
                               layers.
    """
    inception_exclude_scopes = [
        'InceptionV3/AuxLogits', 'InceptionV3/Logits', 'global_step',
        'final_ops'
    ]
    reader = tf.train.NewCheckpointReader(inception_checkpoint_file)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # Get all variables to restore. Exclude Logits and AuxLogits because they
    # depend on the input data and we do not need to intialize them.
    all_vars = tf.contrib.slim.get_variables_to_restore(
        exclude=inception_exclude_scopes)
    # Remove variables that do not exist in the inception checkpoint (for
    # example the final softmax and fully-connected layers).
    inception_vars = {
        var.op.name: var
        for var in all_vars if var.op.name in var_to_shape_map
    }
    inception_saver = tf.train.Saver(inception_vars)
    inception_saver.restore(session, inception_checkpoint_file)

    # Restore the rest of the variables from the trained checkpoint.
    trained_vars = tf.contrib.slim.get_variables_to_restore(
        exclude=inception_exclude_scopes + inception_vars.keys())
    trained_saver = tf.train.Saver(trained_vars)
    trained_saver.restore(session, trained_checkpoint_file)

  def build_prediction_graph(self):
    """Builds prediction graph and registers appropriate endpoints."""

    tensors = self.build_graph(None, 1, GraphMod.PREDICT)

    keys_placeholder = tf.placeholder(tf.string, shape=[None])
    inputs = {
        'key': keys_placeholder,
        'image_bytes': tensors.input_jpeg
    }

    # To extract the id, we need to add the identity function.
    keys = tf.identity(keys_placeholder)
    labels = self.labels + ['UNKNOWN']
    labels_tensor = tf.constant(labels)
    labels_table = tf.contrib.lookup.index_to_string_table_from_tensor(mapping=labels_tensor)
    predicted_label = labels_table.lookup(tensors.predictions[0])

    # Need to duplicate the labels by num_of_instances so the output is one batch
    # (all output members share the same outer dimension).
    # The labels are needed for client to match class scores list.
    labels_tensor = tf.expand_dims(tf.constant(labels), 0)
    num_instance = tf.shape(keys)
    labels_tensors_n = tf.tile(labels_tensor, tf.concat(axis=0, values=[num_instance, [1]]))

    outputs = {
        'key': keys,
        'prediction': predicted_label,
        'labels': labels_tensors_n,
        'scores': tensors.predictions[1],
    }
    return inputs, outputs

  def export(self, last_checkpoint, output_dir):
    """Builds a prediction graph and xports the model.

    Args:
      last_checkpoint: Path to the latest checkpoint file from training.
      output_dir: Path to the folder to be used to output the model.
    """
    logging.info('Exporting prediction graph to %s', output_dir)
    with tf.Session(graph=tf.Graph()) as sess:
      # Build and save prediction meta graph and trained variable values.
      inputs, outputs = self.build_prediction_graph()
      signature_def_map = {
        'serving_default': signature_def_utils.predict_signature_def(inputs, outputs)
      }
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      self.restore_from_checkpoint(sess, self.inception_checkpoint_file,
                                   last_checkpoint)
      init_op_serving = control_flow_ops.group(
          variables.local_variables_initializer(),
          data_flow_ops.tables_initializer())

      builder = saved_model_builder.SavedModelBuilder(output_dir)
      builder.add_meta_graph_and_variables(
          sess, [tag_constants.SERVING],
          signature_def_map=signature_def_map,
          legacy_init_op=init_op_serving)
      builder.save(False)

  def format_metric_values(self, metric_values):
    """Formats metric values - used for logging purpose."""

    # Early in training, metric_values may actually be None.
    loss_str = 'N/A'
    accuracy_str = 'N/A'
    try:
      loss_str = 'loss: %.3f' % metric_values[0]
      accuracy_str = 'accuracy: %.3f' % metric_values[1]
    except (TypeError, IndexError):
      pass

    return '%s, %s' % (loss_str, accuracy_str)

  def format_prediction_values(self, prediction):
    """Formats prediction values - used for writing batch predictions as csv."""
    return '%.3f' % (prediction[0])


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss_op):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(epsilon=0.001)
    train_op = optimizer.minimize(loss_op, global_step)
    return train_op, global_step
