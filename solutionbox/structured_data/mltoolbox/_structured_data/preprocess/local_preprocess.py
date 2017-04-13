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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import argparse
import collections
import csv
import json
import os
import six
import sys


from tensorflow.python.lib.io import file_io


SCHEMA_FILE = 'schema.json'
NUMERICAL_ANALYSIS_FILE = 'stats.json'
CATEGORICAL_ANALYSIS_FILE = 'vocab_%s.csv'


def parse_arguments(argv):
  """Parse command line arguments.

  Args:
    argv: list of command line arguments, includeing programe name.

  Returns:
    An argparse Namespace object.
  """
  parser = argparse.ArgumentParser(
      description='Runs Preprocessing on structured CSV data.')
  parser.add_argument('--input-file-pattern',
                      type=str,
                      required=True,
                      help='Input CSV file names. May contain a file pattern')
  parser.add_argument('--output-dir',
                      type=str,
                      required=True,
                      help='Google Cloud Storage which to place outputs.')
  parser.add_argument('--schema-file',
                      type=str,
                      required=True,
                      help=('BigQuery json schema file'))

  args = parser.parse_args(args=argv[1:])

  # Make sure the output folder exists if local folder.
  file_io.recursive_create_dir(args.output_dir)

  return args


def run_numerical_categorical_analysis(args, schema_list):
  """Makes the numerical and categorical analysis files.

  Args:
    args: the command line args
    schema_list: python object of the schema json file.

  Raises:
    ValueError: if schema contains unknown column types.
  """
  header = [column['name'] for column in schema_list]
  input_files = file_io.get_matching_files(args.input_file_pattern)

  # Check the schema is valid
  for col_schema in schema_list:
    col_type = col_schema['type'].lower()
    if col_type != 'string' and col_type != 'integer' and col_type != 'float':
      raise ValueError('Schema contains an unsupported type %s.' % col_type)

  # initialize the results
  def _init_numerical_results():
    return {'min': float('inf'),
            'max': float('-inf'),
            'count': 0,
            'sum': 0.0}
  numerical_results = collections.defaultdict(_init_numerical_results)
  categorical_results = collections.defaultdict(
      lambda: collections.defaultdict(int))

  # for each file, update the numerical stats from that file, and update the set
  # of unique labels.
  for input_file in input_files:
    with file_io.FileIO(input_file, 'r') as f:
      for line in csv.reader(f):
        parsed_line = dict(zip(header, line))

        for col_schema in schema_list:
          col_name = col_schema['name']
          col_type = col_schema['type']
          if col_type.lower() == 'string':
            split_strings = parsed_line[col_name].split(' ')

            for one_label in split_strings:
              # Filter out empty strings
              if one_label:
                # add the label to the dict and increase its count.
                categorical_results[col_name][one_label] += 1
          else:
            # numerical column.

            # if empty, skip
            if not parsed_line[col_name].strip():
              continue

            numerical_results[col_name]['min'] = (
              min(numerical_results[col_name]['min'],
                  float(parsed_line[col_name])))
            numerical_results[col_name]['max'] = (
              max(numerical_results[col_name]['max'],
                  float(parsed_line[col_name])))
            numerical_results[col_name]['count'] += 1
            numerical_results[col_name]['sum'] += float(parsed_line[col_name])

  # Update numerical_results to just have min/min/mean
  for col_schema in schema_list:
    if col_schema['type'].lower() != 'string':
      col_name = col_schema['name']
      mean = numerical_results[col_name]['sum'] / numerical_results[col_name]['count']
      del numerical_results[col_name]['sum']
      del numerical_results[col_name]['count']
      numerical_results[col_name]['mean'] = mean

  # Write the numerical_results to a json file.
  file_io.write_string_to_file(
      os.path.join(args.output_dir, NUMERICAL_ANALYSIS_FILE),
      json.dumps(numerical_results, indent=2, separators=(',', ': ')))

  # Write the vocab files. Each label is on its own line.
  for name, label_count in six.iteritems(categorical_results):
    # Labels is now the string:
    # label1,count
    # label2,count
    # ...
    # where label1 is the most frequent label, and label2 is the 2nd most, etc.
    labels = '\n'.join(["%s,%d" % (label, count)
                        for label, count in sorted(six.iteritems(label_count),
                                                   key=lambda x: x[1],
                                                   reverse=True)])
    file_io.write_string_to_file(
        os.path.join(args.output_dir, CATEGORICAL_ANALYSIS_FILE % name),
        labels)


def run_analysis(args):
  """Builds an analysis files for training."""

  # Read the schema and input feature types
  schema_list = json.loads(
      file_io.read_file_to_string(args.schema_file).decode())

  run_numerical_categorical_analysis(args, schema_list)

  # Also save a copy of the schema in the output folder.
  file_io.copy(args.schema_file,
               os.path.join(args.output_dir, SCHEMA_FILE),
               overwrite=True)


def main(argv=None):
  args = parse_arguments(sys.argv if argv is None else argv)
  run_analysis(args)


if __name__ == '__main__':
  main()
