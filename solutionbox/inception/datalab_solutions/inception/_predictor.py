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


"""Local implementation for preprocessing, training and prediction for inception model.
"""

import apache_beam as beam
import collections
import google.cloud.ml as ml
import json
import os
import tensorflow as tf

from . import _util


def _tf_predict(model_dir, batches):
  model_dir = os.path.join(model_dir, 'model')
  with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(os.path.join(model_dir, 'export.meta'))
    new_saver.restore(sess, os.path.join(model_dir, 'export'))
    init_op = tf.get_collection(tf.contrib.session_bundle.constants.INIT_OP_KEY)[0]
    sess.run(init_op)
    inputs = json.loads(tf.get_collection('inputs')[0])
    outputs = json.loads(tf.get_collection('outputs')[0])
    for batch in batches:
      feed_dict = collections.defaultdict(list)
      for ii, image_filename in enumerate(batch):
        if image_filename is None:
          break
        with ml.util._file.open_local_or_gcs(image_filename, 'r') as ff:
          image_bytes = ff.read()
          feed_dict[inputs['image_bytes']].append(image_bytes)
          feed_dict[inputs['key']].append(str(ii))
      predictions, labels, scores = sess.run(
          [outputs['prediction'], outputs['labels'], outputs['scores']], feed_dict=feed_dict)
      yield zip(predictions, labels, scores)


def predict(model_dir, image_files):
  """Local instant prediction."""

  # Single batch for instant prediction.
  results = next(_tf_predict(model_dir, [image_files]))
  predicted_and_scores = [(predicted, label_scores[list(labels).index(predicted)])
                          for predicted, labels, label_scores in results]
  return predicted_and_scores


# Helpers for batch prediction dataflow pipeline

class EmitAsBatchDoFn(beam.DoFn):
  """A DoFn that buffers the records and emits them batch by batch."""

  def __init__(self, batch_size):
    self._batch_size = batch_size
    self._cached = []

  def process(self, context):
    element = context.element
    self._cached.append(element)
    if len(self._cached) >= self._batch_size:
      emit = self._cached
      self._cached = []
      yield emit

  def finish_bundle(self, context=None):
    if len(self._cached) > 0:  # pylint: disable=g-explicit-length-test
      yield self._cached


class UnbatchDoFn(beam.DoFn):
  """A DoFn expand batch into elements."""

  def process(self, context):
    for item in context.element:
      yield item


class LoadImagesDoFn(beam.DoFn):
  """A DoFn that reads image from url."""
  
  def process(self, context):
    import google.cloud.ml as ml
    element = context.element  
    with ml.util._file.open_local_or_gcs(element['image_url'], 'r') as ff:
      image_bytes = ff.read()
    out_element = {'image_bytes': image_bytes}
    out_element.update(element)
    yield out_element


class PredictBatchDoFn(beam.DoFn):
  """A DoFn that does batch prediction."""

  def __init__(self, model_dir):
    import os

    self._model_dir = os.path.join(model_dir, 'model')
    self._session = None
    self._tf_inputs = None
    self._tf_outputs = None

  def start_bundle(self, context):
    import json
    import os
    import tensorflow as tf

    self._session = tf.Session()
    new_saver = tf.train.import_meta_graph(os.path.join(self._model_dir, 'export.meta'))
    new_saver.restore(self._session, os.path.join(self._model_dir, 'export'))
    init_op = tf.get_collection(tf.contrib.session_bundle.constants.INIT_OP_KEY)[0]
    self._session.run(init_op)
    self._tf_inputs = json.loads(tf.get_collection('inputs')[0])
    self._tf_outputs = json.loads(tf.get_collection('outputs')[0])

  def finish_bundle(self, context):
    if self._session is not None:
      self._session.close()

  def process(self, context):
    import collections
    
    element = context.element
    image_urls = [x['image_url'] for x in element]
    targets = None
    if 'label' in element[0] and element[0]['label'] is not None:
      targets = [x['label'] for x in element]

    feed_dict = collections.defaultdict(list)
    feed_dict[self._tf_inputs['image_bytes']] = [x['image_bytes'] for x in element]
    feed_dict[self._tf_inputs['key']] = image_urls
    predictions, labels, scores = self._session.run(
        [self._tf_outputs['prediction'], self._tf_outputs['labels'], self._tf_outputs['scores']],
        feed_dict=feed_dict)
    if targets is not None:
      yield zip(image_urls, targets, predictions, labels, scores)
    else:
      yield zip(image_urls, predictions, labels, scores)


class ProcessResultsDoFn(beam.DoFn):
  """A DoFn that process prediction results."""

  def process(self, context):
    element = context.element
    target = None
    if len(element) == 5:
      image_url, target, prediction, labels, scores = element
    else:
      image_url, prediction, labels, scores = element
    labels = list(labels)
    predicted_prob = scores[labels.index(prediction)]
    out_element = {
      'image_url': image_url,
      'predicted': prediction,
      # Convert to float from np.float32 because BigQuery Sink can only handle intrinsic types.
      'predicted_prob': float(predicted_prob)  
    }
    if target is not None:
      target_prob = scores[labels.index(target)] if target in labels else 0.0
      out_element['target_prob'] = float(target_prob)
      out_element['target'] = target
    yield out_element


class MakeCsvLineDoFn(beam.DoFn):
  """A DoFn that makes CSV lines out of prediction results."""

  def process(self, context):
    import csv
    import StringIO

    element = context.element
    line = StringIO.StringIO()
    if len(element) == 5:    
      csv.DictWriter(line, 
          ['image_url', 'target', 'predicted', 'target_prob', 'predicted_prob']).writerow(element)
    else:
      csv.DictWriter(line, ['image_url', 'predicted', 'predicted_prob']).writerow(element)   
    yield line.getvalue()


def configure_pipeline(p, dataset, model_dir, output_csv, output_bq_table):
  """Configures a dataflow pipeline for batch prediction."""

  data = _util.get_sources_from_dataset(p, dataset, 'predict')
  schema = _util.get_schema_from_dataset(dataset)
  if len(schema) > 2 or len(schema) == 0 or any (x['type'] != 'STRING' for x in schema):
    raise Exception('Dataset schema is invalid. Expect one STRING or two STRING columns')
  if len(schema) == 2:
    output_schema = [
        {'name': 'image_url', 'type': 'STRING'},
        {'name': 'target', 'type': 'STRING'},
        {'name': 'predicted', 'type': 'STRING'},
        {'name': 'target_prob', 'type': 'FLOAT'},
        {'name': 'predicted_prob', 'type': 'FLOAT'},
    ]
  else:
    output_schema = [
        {'name': 'image_url', 'type': 'STRING'},
        {'name': 'predicted', 'type': 'STRING'},
        {'name': 'predicted_prob', 'type': 'FLOAT'},
    ]
  results = (data
      | 'Load Images' >> beam.ParDo(LoadImagesDoFn())
      | 'Batch Inputs' >> beam.ParDo(EmitAsBatchDoFn(20))
      | 'Batch Predict' >> beam.ParDo(PredictBatchDoFn(model_dir))
      | 'Unbatch' >> beam.ParDo(UnbatchDoFn())
      | 'Process Results' >> beam.ParDo(ProcessResultsDoFn()))

  if output_csv is not None:
    schema_file = output_csv + '.schema.json'
    results_save = (results
        | 'Prepare For Output' >> beam.ParDo(MakeCsvLineDoFn())
        | 'Write Csv Results' >> beam.io.textio.WriteToText(output_csv, shard_name_template=''))
    (results_save 
        | beam.transforms.combiners.Sample.FixedSizeGlobally('Sample One', 1)
        | 'Serialize Schema' >> beam.Map(lambda path: json.dumps(output_schema))
        | 'Write Schema' >> beam.io.textio.WriteToText(schema_file, shard_name_template=''))
  if output_bq_table is not None:
    # BigQuery sink takes schema in the form of 'field1:type1,field2:type2...'
    bq_schema_string = ','.join(x['name'] + ':' + x['type'] for x in output_schema)
    sink = beam.io.BigQuerySink(output_bq_table, schema=bq_schema_string,
        write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE)
    results | 'Write BQ Results' >> beam.io.Write(sink)
  
