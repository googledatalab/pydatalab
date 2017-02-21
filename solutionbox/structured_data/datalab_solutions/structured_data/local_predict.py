import argparse
import collections
import json
import os
import sys

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.contrib.session_bundle import bundle_shim  


  

# copy + edit from /google3/third_party/py/googlecloudsdk/command_lib/ml/predict_lib_beta.py
def _get_signature_from_meta_graph(meta_graph, named_key=None):
  """Returns the SignatureDef in meta_graph update dtypes using graph."""
  if not meta_graph.signature_def:
    raise Exception("MetaGraph must have at least one signature_def.")
  #named_key = "serving_default_from_named"

  print('The graph has the following signatures for serving.')
  for name, key in meta_graph.signature_def.iteritems():
    print(name)

  print('namedkey=', named_key)
  print('glcoud will try to use serving_default_from_named first if it exist'
        'otherwise, serving_default better be in the signature_def!!!')

  if named_key in meta_graph.signature_def:
    return meta_graph.signature_def[named_key]

  # TODO(b/34690042): document these and point to a public, canonical constant.
  signature = meta_graph.signature_def["serving_default"]


  return signature



def local_predict(args):
  """Runs prediction locally."""
  print(args)

  session, meta_graph = bundle_shim.load_session_bundle_or_saved_model_bundle_from_path(args.model_dir, tags=[tag_constants.SERVING])
  signature = _get_signature_from_meta_graph(meta_graph, named_key=args.graph_head)

  #print(session)
  #print(meta_graph)
  #print(signature)

  # back to the normal local_predict code

  # get the mappings between aliases and tensor names
  # for both inputs and outputs
  input_alias_map = {friendly_name: tensor_info_proto.name 
      for (friendly_name, tensor_info_proto) in signature.inputs.items() }
  output_alias_map = {friendly_name: tensor_info_proto.name 
      for (friendly_name, tensor_info_proto) in signature.outputs.items() }
  aliases, tensor_names = zip(*output_alias_map.items())

  print('\ninput_alias_map ' + str(input_alias_map))
  print('\noutput_alias_map '+ str(output_alias_map))

  batch_size = 2


  # Don't predict the whole file, just the first batch_size many. 
  with open(args.input_data[0]) as f:
    feed_dict = get_feed_dict(f, args, input_alias_map, output_alias_map, aliases, tensor_names)

    print('feed_dict', feed_dict)
  
    # run the graph.
    result = session.run(fetches=tensor_names,
                         feed_dict=feed_dict)

    print('result ' + str(result))
    for row in zip(*result):
       print json.dumps({
           name: (value.tolist() if getattr(value, 'tolist', None) else
                  value)
           for name, value in zip(aliases, row)
       })
  
def get_feed_dict(f, args, input_alias_map, output_alias_map, aliases, tensor_names):
  feed_dict = collections.defaultdict(list)
  batch_size = 2
  for i in range(batch_size):
    if args.graph_head == 'json':
      json_line = json.loads(f.readline())
      for friendly_name, tensor_name in input_alias_map.iteritems():
        # What do I do with missing values???
        feed_dict[tensor_name].append(json_line[friendly_name])
    elif args.graph_head == 'csv':
      csv_line = f.readline()
      assert(len(input_alias_map) == 1)
      friendly_name = input_alias_map.keys()[0]  
      tensor_name = input_alias_map.values()[0]
      feed_dict[tensor_name].append(csv_line)
    else:
      raise ValueError('ops in get_feed_dict')
  
  return feed_dict
  
def get_args():
  """Parses arguments specified on the command-line."""

  parser = argparse.ArgumentParser()

  parser.add_argument(
      'input_data',
      nargs='+',
      help=('The input data file. Multiple files can be specified if more than '
            'one file is needed.'))

  parser.add_argument(
      '--graph_head',
      type=str,
      default='json')
      
  parser.add_argument(
      '--model_dir',
      dest='model_dir',
      help=('The path to the model where the tensorflow meta graph '
            'proto and checkpoint files are saved.'))
  args = parser.parse_args()


  return args

if __name__ == '__main__':
  args = get_args()
  local_predict(args)