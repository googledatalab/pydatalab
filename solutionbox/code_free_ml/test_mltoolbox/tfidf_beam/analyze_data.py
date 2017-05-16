# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Criteo Classification Sample Preprocessing Runner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import subprocess
import sys


import apache_beam as beam
import tensorflow as tf

from tensorflow_transform import coders

import tensorflow_transform as tft
from tensorflow_transform.beam import impl as tft_impl
from tensorflow_transform.beam import tft_beam_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform import api
from tensorflow_transform import analyzers
from tensorflow.contrib import lookup


def parse_arguments(argv):
  """Parse command line arguments.

  Args:
    argv: list of command line arguments including program name.
  Returns:
    The parsed arguments as returned by argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser(
      description='Runs Preprocessing on the Criteo model data.')

  parser.add_argument(
      '--training_data',
      default='./train.csv')
  parser.add_argument(
      '--eval_data',
      default='./eval.csv')
  parser.add_argument(
      '--predict_data', help='Data to encode as prediction features.')
  parser.add_argument(
      '--output_dir',
      default='aout',
      help=('Google Cloud Storage or Local directory in which '
            'to place outputs.'))
  args, _ = parser.parse_known_args(args=argv[1:])


  return args


def make_input_schema():
  result = {
    'target': tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
    'key': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0),
    'str_tfidf': tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
  } 
  return dataset_schema.from_feature_spec(result)

def make_coder(schema):
  column_names = [
    'key', 'target', 'str_tfidf'
  ]
  return coders.CsvCoder(column_names, schema, delimiter=',')


def make_preprocessing_fn():
  def preprocessing_fn(inputs):
    inputs_as_ints = tft.string_to_int(tft.map(tf.string_split, inputs['str_tfidf']))
    result = {
        'key': inputs['key'],
        'target': tft.string_to_int(inputs['str_tfidf']),
        'str_tfidf_ids': inputs_as_ints,
        'str_tfidf_weights': tft.tfidf_weights(inputs_as_ints, 6+1),
    }
    return result

  return preprocessing_fn


def preprocess(pipeline, training_data, eval_data, output_dir):
  """Run pre-processing step as a pipeline.

  Args:
    pipeline: beam pipeline
    training_data: file paths to input csv files.
    eval_data: file paths to input csv files.
    predict_data: file paths to input csv files.
    output_dir: file path to where to write all the output files.
    frequency_threshold: frequency threshold to use for categorical values.
  """
  input_schema = make_input_schema()
  coder = make_coder(input_schema)

  # 3) Read from text using the coder.
  train_data = (
      pipeline
      | 'ReadTrainingData' >> beam.io.ReadFromText(training_data)
      | 'ParseTrainingCsv' >> beam.Map(coder.decode))

  evaluate_data = (
      pipeline
      | 'ReadEvalData' >> beam.io.ReadFromText(eval_data)
      | 'ParseEvalCsv' >> beam.Map(coder.decode))

  input_metadata = dataset_metadata.DatasetMetadata(schema=input_schema)
  _ = (input_metadata
       | 'WriteInputMetadata' >> tft_beam_io.WriteMetadata(
           os.path.join(output_dir, 'raw_metadata'),
           pipeline=pipeline))


  preprocessing_fn = make_preprocessing_fn()
  (train_dataset, train_metadata), transform_fn = (
      (train_data, input_metadata)
      | 'AnalyzeAndTransform' >> tft_impl.AnalyzeAndTransformDataset(
          preprocessing_fn))

  metadata_io.write_metadata(
       metadata=train_metadata,
       path=os.path.join(output_dir, 'bad_training_metadata'))

  _ = (transform_fn | 'WriteTransformFn' >> tft_beam_io.WriteTransformFn(output_dir))


  (evaluate_dataset, evaluate_metadata) = (
      ((evaluate_data, input_metadata), transform_fn)
      | 'TransformEval' >> tft_impl.TransformDataset())

  train_coder = coders.ExampleProtoCoder(train_metadata.schema)
  _ = (train_dataset
       | 'SerializeTrainExamples' >> beam.Map(train_coder.encode)
       | 'WriteTraining'
       >> beam.io.WriteToTFRecord(
           os.path.join('exout',
                        'ftrain'),
           file_name_suffix='.tfrecord.gz'))

  evaluate_coder = coders.ExampleProtoCoder(evaluate_metadata.schema)
  _ = (evaluate_dataset
       | 'SerializeEvalExamples' >> beam.Map(evaluate_coder.encode)
       | 'WriteEval'
       >> beam.io.WriteToTFRecord(
           os.path.join('exout',
                        'feval'),
           file_name_suffix='.tfrecord.gz'))

def main(argv=None):
  """Run Preprocessing as a Dataflow."""
  args = parse_arguments(sys.argv if argv is None else argv)
  pipeline_name = 'DirectRunner'
  pipeline_options = None

  temp_dir = os.path.join(args.output_dir, 'tmp')
  with beam.Pipeline(pipeline_name, options=pipeline_options) as p:
    with tft_impl.Context(temp_dir=temp_dir):
      preprocess(
          pipeline=p,
          training_data=args.training_data,
          eval_data=args.eval_data,
          output_dir=args.output_dir)


if __name__ == '__main__':
  main()