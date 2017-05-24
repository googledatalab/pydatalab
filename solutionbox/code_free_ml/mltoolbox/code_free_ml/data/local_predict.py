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
import os
import six
import sys
import json

import tensorflow as tf
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
                      help='One prediction output file is made per input file.')
  parser.add_argument('--no-shard-files',
                      dest='shard_files',
                      action='store_false',
                      help='Don\'t shard files')
  parser.set_defaults(shard_files=True)

  args, _ = parser.parse_known_args(args=argv[1:])

  return args


def batch_csv_reader(csv_file, batch_size):
  """Reads a csv file in batches.

  Args:
    csv_file: file path
    batch_size: positive integer

  Returns:
    generator
  """
  counter = 0
  batched_lines = []
  for one_csv_line in file_io.FileIO(csv_file, 'r'):
    batched_lines.append(one_csv_line.rstrip())
    counter += 1
    if counter == batch_size:
      yield batched_lines
      counter = 0
      batched_lines = []

  if batched_lines:
    yield batched_lines


def run_batch_prediction(session, input_alias_map, output_alias_map, batch_csv_strings):
  """Runs a tensorflow prediction graph from batched csv input.

  Args:
    session: tensorflow session
    input_alias_map: dict of {friendly_name: tensor_name} for feeding the
        feed_dict. Should only have 1 value as the model takes csv data.
    output_alias_map: dict of {friendly_name: tensor_name} for fetching the
        ouput tensors form session.run
    batch_csv_strings: list of csv strings to feed to the model.

  Returns:
    List of dicts in the form {output_alias_friendly_name: predicted_value}.
  """
  _, csv_tensor_name = input_alias_map.items()[0]
  raw_result = session.run(fetches=output_alias_map,
                           feed_dict={csv_tensor_name: batch_csv_strings})

  # Convert from numpy arrays to python lists
  result = {key: raw_result[key].flatten().tolist() for key in raw_result}

  batch_size = len(batch_csv_strings)
  json_list = []
  for i in range(batch_size):
    json_result = {key: result[key][i] for key in result}
    json_list.append(json_result)

  return json_list


def format_results(output_format, output_schema, batched_json_results):
  """Formats prediction results.

  Args:
    output_format: 'csv' or 'json'
    output_schema: output file schema. Only used if output_format is csv. This
        is where the csv column order is defined.
    batched_json_results: list of dicts. The output of run_batch_prediction().

  Returns:
    List of csv lines or a list of json strings.
  """
  if output_format == 'csv':
    batched_results = []
    for one_json_result in batched_json_results:
      values = [str(one_json_result[schema['name']])
                for schema in output_schema]
      batched_results.append(','.join(values))
  elif output_format == 'json':
    batched_results = [json.dumps(x) for x in batched_json_results]
  else:
    raise ValueError('Unknown output_format %s' % output_format)

  return batched_results


def write_schema_file(output_location, output_schema):
  """Writes the schema file."""
  file_io.write_string_to_file(
      os.path.join(output_location, 'schema.json'),
      json.dumps(output_schema, indent=2))


def append_batched_strings(file_path, batched_formatted_strings):
  """Appends a list of strings to a text file."""
  with file_io.FileIO(file_path, 'a') as f:
    for line in batched_formatted_strings:
      f.write(line + '\n')


def get_output_schema(session, output_alias_map):
  """Makes output schema.

  Args:
    session: tensorflow session
    output_alias_map: dict of {friendly_name: tensor_name}. The session's graph
        is searched for the tensor's datatype.

  Returns:
    A bigquery-stype schema object.
  """
  schema = []
  for name in sorted(six.iterkeys(output_alias_map)):
    tensor_name = output_alias_map[name]
    dtype = session.graph.get_tensor_by_name(tensor_name).dtype
    if dtype == tf.int32 or dtype == tf.int64:
      schema.append({'name': name, 'type': 'INTEGER'})
    elif dtype == tf.float32 or dtype == tf.float64:
      schema.append({'name': name, 'type': 'FLOAT'})
    else:
      schema.append({'name': name, 'type': 'STRING'})

  return schema


def main(argv=None):
  """Runs local batch prediction."""
  args = parse_arguments(sys.argv if argv is None else argv)

  file_io.recursive_create_dir(args.output_location)
  with tf.Graph().as_default(), tf.Session() as sess:
    # Load the graph for prediction
    meta_graph_pb = tf.saved_model.loader.load(
        sess=sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        export_dir=args.trained_model_dir)
    signature = meta_graph_pb.signature_def['serving_default']

    input_alias_map = {
        friendly_name: tensor_info_proto.name
        for (friendly_name, tensor_info_proto) in signature.inputs.items()}
    output_alias_map = {
        friendly_name: tensor_info_proto.name
        for (friendly_name, tensor_info_proto) in signature.outputs.items()}

    output_schema = get_output_schema(sess, output_alias_map)

    if 1 != len(input_alias_map.keys()):
      raise ValueError('Graph has more than one input placeholder')

    # Loop over all the files, read them in batches, and run prediction.
    csv_files = file_io.get_matching_files(args.predict_data)
    for file_num, csv_file in enumerate(csv_files):
      for batch in batch_csv_reader(csv_file, args.batch_size):
        # Run prediction, and format the output as csv or json
        batched_json_results = run_batch_prediction(
            session=sess,
            input_alias_map=input_alias_map,
            output_alias_map=output_alias_map,
            batch_csv_strings=batch)
        batched_formatted_results = format_results(
            args.output_format, output_schema, batched_json_results)

        # write the output file(s)
        if args.shard_files:
          file_base_name = 'predictions-{index:05d}-of-{count:05d}'.format(
              index=file_num, count=len(csv_files))
        else:
          file_base_name = 'predictions-00000-of-00001'.format(
              index=file_num, count=len(csv_files))

        if args.output_format == 'csv':
          file_name = file_base_name + '.csv'
        elif args.output_format == 'json':
          file_name = file_base_name + '.json'

        file_path = os.path.join(args.output_location, file_name)
        append_batched_strings(file_path, batched_formatted_results)

  # Write the schema file
  write_schema_file(args.output_location, output_schema)


if __name__ == '__main__':
  main()
