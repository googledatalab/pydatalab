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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pandas as pd
import re
import six
import sys
import math
import multiprocessing
import json


import tensorflow as tf
from tensorflow.python.framework import dtypes

from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.lib.io import file_io

from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import metadata_io

from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_schema


from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.contrib.learn.python.learn import export_strategy
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)

from tensorflow.python.ops import variables
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.python.training import saver
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.client import session as tf_session
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import compat
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_def_utils



# Files
TRANSFORMED_METADATA_DIR = 'transformed_metadata'
RAW_METADATA_DIR = 'raw_metadata'
TRANSFORM_FN_DIR = 'transform_fn'



# Input file columns
KEY_COL = 'key'
TARGET_COL = 'target'
TEXT_COL = 'str_tfidf'

def parse_arguments(argv):
  """Parse the command line arguments."""
  parser = argparse.ArgumentParser(
      description=('Train a regression or classification model. Note that if '
                   'using a DNN model, --layer-size1=NUM, --layer-size2=NUM, '
                   'should be used. '))

  # I/O file parameters
  parser.add_argument('--train-data-paths', type=str,
                      default='./exout/ftrain*')
  parser.add_argument('--eval-data-paths', type=str,
                      default='./exout/feval*')
  parser.add_argument('--job-dir', type=str,
                      default='tout')
  parser.add_argument('--analysis-output-dir',
                      type=str,
                      default='./aout')

  # HP parameters
  parser.add_argument('--learning-rate', type=float, default=0.01,
                      help='tf.train.AdamOptimizer learning rate')
  parser.add_argument('--epsilon', type=float, default=0.0005,
                      help='tf.train.AdamOptimizer epsilon')

  # Training input parameters
  parser.add_argument('--max-steps', type=int, default=1000,
                      help='Maximum number of training steps to perform.')
  parser.add_argument('--num-epochs',
                      type=int,
                      help=('Maximum number of training data epochs on which '
                            'to train. If both --max-steps and --num-epochs '
                            'are specified, the training job will run for '
                            '--max-steps or --num-epochs, whichever occurs '
                            'first. If unspecified will run for --max-steps.'))
  parser.add_argument('--train-batch-size', type=int, default=100)
  parser.add_argument('--eval-batch-size', type=int, default=50)
  parser.add_argument('--min-eval-frequency', type=int, default=100,
                      help=('Minimum number of training steps between '
                            'evaluations'))

  args, remaining_args = parser.parse_known_args(args=argv[1:])

  return args

def build_feature_columns():
  feature_columns = []

  sparse_ids = tf.contrib.layers.sparse_column_with_integerized_feature(
      'str_tfidf_ids',
      bucket_size=6+1)
  sparse_weights =  tf.contrib.layers.weighted_sparse_column(
      sparse_id_column=sparse_ids, 
      weight_column_name='str_tfidf_weights',
      )

  new_feature = tf.contrib.layers.one_hot_column(sparse_weights)
  #new_feature = tf.contrib.layers.one_hot_column(sparse_ids)
  #new_feature = tf.contrib.layers.embedding_column(sparse_weights, dimension=2)
  #new_feature = tf.contrib.layers.embedding_column(sparse_ids, dimension=2)
  #new_feature = sparse_weights
  #new_feature = sparse_ids

  feature_columns.append(new_feature)
  return feature_columns



def get_estimator(args):
 
  # Build tf.learn features
  feature_columns = build_feature_columns()

  # Set how often to run checkpointing in terms of time.
  config = tf.contrib.learn.RunConfig(
      save_checkpoints_secs=600)

  train_dir = os.path.join(args.job_dir, 'train')
  estimator = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 5],
        n_classes=3,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.AdamOptimizer(
            args.learning_rate, epsilon=args.epsilon))
  #estimator = tf.contrib.learn.LinearClassifier(
  #      feature_columns=feature_columns,
  #      n_classes=3,
  #      config=config,
  #      model_dir=train_dir,
  #      optimizer=tf.train.AdamOptimizer(
  #          args.learning_rate, epsilon=args.epsilon))
  return estimator

def gzip_reader_fn():
  return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
      compression_type=tf.python_io.TFRecordCompressionType.GZIP))

def get_experiment_fn(args):
  """Builds the experiment function for learn_runner.run.

  Args:
    args: the command line args

  Returns:
    A function that returns a tf.learn experiment object.
  """

  def get_experiment(output_dir):
    estimator = get_estimator(args)

    # Build readers for training.
    transformed_metadata = metadata_io.read_metadata(
        os.path.join(args.analysis_output_dir, TRANSFORMED_METADATA_DIR))
    raw_metadata = metadata_io.read_metadata(
        os.path.join(args.analysis_output_dir, RAW_METADATA_DIR))

    export_strategy = make_export_strategy(args)

    input_reader_for_train = input_fn_maker.build_training_input_fn(
        metadata=transformed_metadata,
        file_pattern=args.train_data_paths,
        training_batch_size=args.train_batch_size,
        reader=gzip_reader_fn,
        label_keys=[TARGET_COL],
        feature_keys=None,  # extract all features
        key_feature_name=None,  # None as we take care of the key column.
        reader_num_threads=multiprocessing.cpu_count(),
        queue_capacity=args.train_batch_size * multiprocessing.cpu_count() + 10,
        randomize_input=True,
        num_epochs=args.num_epochs,
    )
    input_reader_for_eval = input_fn_maker.build_training_input_fn(
        metadata=transformed_metadata,
        file_pattern=args.eval_data_paths,
        training_batch_size=args.eval_batch_size,
        reader=gzip_reader_fn,
        label_keys=[TARGET_COL],
        feature_keys=None,  # extract all features
        key_feature_name=None,  # None as we take care of the key column.
        reader_num_threads=multiprocessing.cpu_count(),
        queue_capacity=args.train_batch_size * multiprocessing.cpu_count() + 10,
        randomize_input=False,
        num_epochs=1,
    )

    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=input_reader_for_train,
        eval_input_fn=input_reader_for_eval,
        train_steps=args.max_steps,
        export_strategies=[export_strategy],
        min_eval_frequency=args.min_eval_frequency,
        eval_steps=None,
    )

  # Return a function to create an Experiment.
  return get_experiment


def make_export_strategy(args):
  """Makes prediction graph.

  Args:
    args: command line args
  """
  raw_metadata = raw_metadata = metadata_io.read_metadata(
        os.path.join(args.analysis_output_dir, RAW_METADATA_DIR))


  def export_fn(estimator, export_dir_base, checkpoint_path=None, eval_result=None):
    with ops.Graph().as_default() as g:
      contrib_variables.create_global_step(g)

      # Build each signature def graph.
      signature_def_map = {}
      input_ops = input_fn_maker.build_default_transforming_serving_input_fn(
              raw_metadata=raw_metadata,
              transform_savedmodel_dir=os.path.join(args.analysis_output_dir, TRANSFORM_FN_DIR),
              raw_label_keys=[TARGET_COL],
              raw_feature_keys=[TARGET_COL, KEY_COL, TEXT_COL],
              convert_scalars_to_vectors=True)()    

      print('export fn ' + '1'*100)
      print(input_ops)
      for f in input_ops.features:
        print(f, input_ops.features[f].get_shape())   
      model_fn_ops = estimator._call_model_fn(input_ops.features,
                                                None,
                                                model_fn_lib.ModeKeys.INFER)
      signature_def_map.update({
          'serving_default': signature_def_utils.predict_signature_def(input_ops.default_inputs,
                                                                    model_fn_ops.predictions)
      })

      if not checkpoint_path:
        # Locate the latest checkpoint
        checkpoint_path = saver.latest_checkpoint(estimator._model_dir)
      if not checkpoint_path:
        raise NotFittedError("Couldn't find trained model at %s."
                             % estimator._model_dir)

      export_dir = saved_model_export_utils.get_timestamped_export_dir(
          export_dir_base)

      with tf_session.Session('') as session:
        # variables.initialize_local_variables()
        variables.local_variables_initializer()
        data_flow_ops.tables_initializer()
        saver_for_restore = saver.Saver(
            variables.global_variables(),
            sharded=True)
        saver_for_restore.restore(session, checkpoint_path)

        init_op = control_flow_ops.group(
            variables.local_variables_initializer(),
            data_flow_ops.tables_initializer())

        # Perform the export
        builder = saved_model_builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(
            session, [tag_constants.SERVING],
            signature_def_map=signature_def_map,
            assets_collection=ops.get_collection(
                ops.GraphKeys.ASSET_FILEPATHS),
            legacy_init_op=init_op)
        builder.save(False)

    # only keep the last 3 models
    saved_model_export_utils.garbage_collect_exports(
        export_dir_base,
        exports_to_keep=3)

    return export_dir

  intermediate_dir = 'intermediate_models'

  return export_strategy.ExportStrategy(intermediate_dir, export_fn)



def main(argv=None):
  """Run a Tensorflow model on the Iris dataset."""
  args = parse_arguments(sys.argv if argv is None else argv)

  tf.logging.set_verbosity(tf.logging.INFO)
  learn_runner.run(
      experiment_fn=get_experiment_fn(args),
      output_dir=args.job_dir)


if __name__ == '__main__':
  main()
