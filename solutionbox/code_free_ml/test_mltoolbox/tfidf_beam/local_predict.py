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

  if named_key in meta_graph.signature_def:
    return meta_graph.signature_def[named_key]

  # TODO(b/34690042): document these and point to a public, canonical constant.
  signature = meta_graph.signature_def["serving_default"]


  return signature



def local_predict(args):
  """Runs prediction locally."""
  print(args)

  session, meta_graph = bundle_shim.load_session_bundle_or_saved_model_bundle_from_path(args.model_dir, tags=[tag_constants.SERVING])
  signature = _get_signature_from_meta_graph(meta_graph, named_key=None)

  input_alias_map = {friendly_name: tensor_info_proto.name 
      for (friendly_name, tensor_info_proto) in signature.inputs.items() }
  output_alias_map = {friendly_name: tensor_info_proto.name 
      for (friendly_name, tensor_info_proto) in signature.outputs.items() }
  print('\ninput_alias_map ' + str(input_alias_map))
  print('\noutput_alias_map '+ str(output_alias_map))      
  aliases, tensor_names = zip(*output_alias_map.items())


  batch_size = 2


  # Don't predict the whole file, just the first batch_size many. 
  #with open(args.input_data[0]) as f:

  feed_dict = {'key': [12,11],
               'str_tfidf': ['bike train car', 'done pizzzzza'],
               'target': ['101', '100']}
  feed_placeholders = {}
  for key in feed_dict:
    feed_placeholders[input_alias_map[key]] = feed_dict[key]


  print('feed_dict', feed_placeholders)

  # run the graph.
  result = session.run(fetches=tensor_names,
                       feed_dict=feed_placeholders)

  print('result ' + str(result))
  for row in zip(*result):
     print json.dumps({
         name: (value.tolist() if getattr(value, 'tolist', None) else
                value)
         for name, value in zip(aliases, row)
     })
  
  
def get_args():
  """Parses arguments specified on the command-line."""

  parser = argparse.ArgumentParser()

      
  parser.add_argument(
      '--model-dir',
      dest='model_dir',
      help=('The path to the model where the tensorflow meta graph '
            'proto and checkpoint files are saved.'))
  args = parser.parse_args()


  return args

if __name__ == '__main__':
  args = get_args()
  local_predict(args)