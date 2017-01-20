
import argparse
import collections
import json
import os
import subprocess
import tensorflow as tf

from tensorflow.python.lib.io import tf_record
from google.cloud.ml import session_bundle


def local_predict(args):
  """Runs prediction locally."""
  print(args)
  session, _ = session_bundle.load_session_bundle_from_path(args.model_dir)
  # get the mappings between aliases and tensor names
  # for both inputs and outputs
  print(session.graph.get_collection('inputs'))
  print(session.graph.get_collection('outputs'))
  print('io collections')

  input_alias_map = json.loads(session.graph.get_collection('inputs')[0])
  output_alias_map = json.loads(session.graph.get_collection('outputs')[0])
  aliases, tensor_names = zip(*output_alias_map.items())

  for input_file in args.input:
    feed_dict = collections.defaultdict(list)
    print('reading ', input_file)
    for line in tf_record.tf_record_iterator(input_file):
      ex = tf.train.Example()
      ex.ParseFromString(line)
      print(ex)
      feed_dict = collections.defaultdict(list)
      feed_dict[input_alias_map['input_example']].append(line)
      #print(tensor_names, feed_dict)
      result = session.run(fetches=tensor_names, feed_dict=feed_dict)
      for row in zip(*result):
        print json.dumps(
          {name: (value.tolist() if getattr(value, 'tolist', None) else value)
          for name, value in zip(aliases, row)})


def parse_args():
  """Parses arguments specified on the command-line."""

  argparser = argparse.ArgumentParser('Predict locally')

  argparser.add_argument(
      'input',
      nargs='+',
      help=('The input data file/file patterns. Multiple '
            'files can be specified if more than one file patterns is needed.'))

  argparser.add_argument(
      '--model_dir',
      dest='model_dir',
      help=('The path to the model where the tensorflow meta graph '
            'proto and checkpoint files are saved.'))

  return argparser.parse_args()


if __name__ == '__main__':
  arguments = parse_args()
  local_predict(arguments)