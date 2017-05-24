# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Runs local prediction on a trained model."""


import argparse
import datetime
import os
import sys

from tensorflow.python.lib.io import file_io


def parse_arguments(argv):
  """Parse command line arguments.
  Args:
    argv: includes the script's name.
  Returns:
    argparse object
  """
  parser = argparse.ArgumentParser(
      description='Runs local prediction')
  # I/O args
  parser.add_argument('--predict-data',
                      required=True,
                      help='Data to run prediction on')
  parser.add_argument('--trained-model-dir',
                      required=True,
                      help='Usually train_output_path/model.')
  parser.add_argument('--output-location',
                      required=True,
                      help=('Location to save output.'))
  parser.add_argument('--output-format',
                      default='csv',
                      choices=['csv', 'json'],
                      help=('format of prediction results.'))

  # Other args
  parser.add_argument('--batch-size',
                      required=False,
                      default=1000,
                      type=int,
                      help=('Batch size. Larger values consumes more memrory '
                            'but takes less time to finish.'))
  parser.add_argument('--shard-files',
                      dest='shard_files',
                      action='store_true',
                      help='Shard files')
  parser.add_argument('--no-shard-files',
                      dest='shard_files',
                      action='store_false',
                      help='Don\'t shard files')
  parser.set_defaults(shard_files=True)


  args, _ = parser.parse_known_args(args=argv[1:])

  return args

def csv_reader(predict_data):
  csv_files = file_io.get_matching_files(predict_data)

  for csv_file in csv_files:
    for csv_line in file_io.FileIO(csv_file, 'r'):
      yield csv_line.rstrip()

def batch_csv_reader(predict_data, batch_size):

  counter = 0
  batched_lines = []
  for one_csv_line in csv_reader(predict_data):
    batched_lines.append(one_csv_line)
    counter += 1
    if counter == batch_size:
      yield batched_lines 
      counter = 0
      batched_lines = []

  if batched_lines:
    yield batched_lines


def main(argv=None):
  args = parse_arguments(sys.argv if argv is None else argv)

  for batch in batch_csv_reader(args.predict_data, args.batch_size):
    print(len(batch))
    print(batch)


if __name__ == '__main__':
  main()

