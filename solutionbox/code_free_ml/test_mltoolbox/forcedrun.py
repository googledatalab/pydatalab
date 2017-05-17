from __future__ import absolute_import

import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import tensorflow as tf
from tensorflow.python.lib.io import file_io


problem_type = 'regression'
model_type = 'dnn'
has_target = False
   
if has_target:
  model_path = './tmp/train_output/evaluation_model'
else:
  model_path = './tmp/train_output/model'

sess = tf.Session()
meta_graph_pb = tf.saved_model.loader.load(
    sess=sess,
    tags=[tf.saved_model.tag_constants.SERVING],
    export_dir=model_path)
signature = meta_graph_pb.signature_def['serving_default']

input_alias_map = {friendly_name: tensor_info_proto.name 
    for (friendly_name, tensor_info_proto) in signature.inputs.items() }
output_alias_map = {friendly_name: tensor_info_proto.name 
    for (friendly_name, tensor_info_proto) in signature.outputs.items() }

feed_dict = {'key': [12,11],
             'target': [-49, -9,] if problem_type == 'regression' else ['100', '101'],
             'num_id': [11, 10],
             'num_scale': [22.29, 5.20],
             'str_one_hot': ['brown', ''],
             'str_embedding': ['def', 'def'],
             'str_bow': ['drone', 'drone truck bike truck'],
             'str_tfidf': ['bike train train car', 'train']
}
if not has_target:
  del feed_dict['target']

expected_output_keys = ['predicted', 'key']
if has_target:
  expected_output_keys.append('target')
if problem_type == 'classification':
  expected_output_keys.extend(['score', 'score_2', 'score_3', 'predicted_2', 'predicted_3'])

feed_placeholders = {}
for key in input_alias_map:
  feed_placeholders[input_alias_map[key]] = feed_dict[key]

output_alias_map = {u'predicted': u'Squeeze_1:0', u'key': u'Squeeze:0'}
feed_placeholders = {
u'Placeholder_4:0': ['brown', 'brown'], 
u'Placeholder_6:0': [22.29, 5.2], 
u'Placeholder_1:0': ['drone', 'drone truck bike truck'], 
u'Placeholder_7:0': ['bike train train car', 'train'], 
u'Placeholder_2:0': [11, 10], 
u'Placeholder_3:0': ['def', 'def'], 
u'Placeholder_5:0': [12, 11]}




result = sess.run(fetches=output_alias_map,
                  feed_dict=feed_placeholders)

print(result)
