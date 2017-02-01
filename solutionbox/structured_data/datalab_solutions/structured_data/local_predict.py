
import argparse
import collections
import json
import os
import sys

from google.cloud.ml import features
from google.cloud.ml import session_bundle


def local_predict(input_data, model_dir):
  """Runs prediction locally."""

  session, _ = session_bundle.load_session_bundle_from_path(model_dir)
  # get the mappings between aliases and tensor names
  # for both inputs and outputs

  print('input map', session.graph.get_collection('inputs'))
  print('output map', session.graph.get_collection('outputs'))

  input_alias_map = json.loads(session.graph.get_collection('inputs')[0])
  output_alias_map = json.loads(session.graph.get_collection('outputs')[0])
  aliases, tensor_names = zip(*output_alias_map.items())

  for input_file in input_data:
    with open(input_file) as f:
      feed_dict = collections.defaultdict(list)
      counter = 0
      for line in f:
        if counter > 4:
          break
        counter += 1
        line = line.strip()
        print('.'+line+'.')
        feed_dict[input_alias_map.values()[0]].append(line)
      result = session.run(fetches=tensor_names, feed_dict=feed_dict)
      for row in zip(*result):
        print json.dumps({
            name: (value.tolist() if getattr(value, 'tolist', None) else
                   value)
            for name, value in zip(aliases, row)
        })


def parse_args(args):
  """Parses arguments specified on the command-line."""

  argparser = argparse.ArgumentParser('Predict on the Iris model.')

  argparser.add_argument(
      'input_data',
      nargs='+',
      help=('The input data file. Multiple files can be specified if more than '
            'one file is needed.'))

  argparser.add_argument(
      '--model_dir',
      dest='model_dir',
      help=('The path to the model where the tensorflow meta graph '
            'proto and checkpoint files are saved.'))

  return argparser.parse_args(args)

if __name__ == '__main__':
  parsed_args = parse_args(sys.argv[1:])
  local_predict(**vars(parsed_args))