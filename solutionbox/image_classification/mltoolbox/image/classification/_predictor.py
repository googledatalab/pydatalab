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
import json
import os

from . import _util


def _load_tf_model(model_dir):
  from tensorflow.python.saved_model import tag_constants
  from tensorflow.contrib.session_bundle import bundle_shim

  model_dir = os.path.join(model_dir, 'model')
  session, meta_graph = bundle_shim.load_session_bundle_or_saved_model_bundle_from_path(
      model_dir, tags=[tag_constants.SERVING])
  signature = meta_graph.signature_def['serving_default']
  inputs = {friendly_name: tensor_info_proto.name
            for (friendly_name, tensor_info_proto) in signature.inputs.items()}
  outputs = {friendly_name: tensor_info_proto.name
             for (friendly_name, tensor_info_proto) in signature.outputs.items()}
  return session, inputs, outputs


def _tf_predict(model_dir, images):
  session, inputs, outputs = _load_tf_model(model_dir)

  with session:
    feed_dict = collections.defaultdict(list)
    for ii, image in enumerate(images):
      feed_dict[inputs['image_bytes']].append(image)
      feed_dict[inputs['key']].append(str(ii))
    predictions, labels, scores = session.run(
      [outputs['prediction'], outputs['labels'], outputs['scores']], feed_dict=feed_dict)

  return zip(predictions, labels, scores)


def predict(model_dir, images):
  """Local instant prediction."""

  results = _tf_predict(model_dir, images)
  predicted_and_scores = [(predicted, label_scores[list(labels).index(predicted)])
                          for predicted, labels, label_scores in results]
  return predicted_and_scores


# Helpers for batch prediction dataflow pipeline

class EmitAsBatchDoFn(beam.DoFn):
  """A DoFn that buffers the records and emits them batch by batch."""

  def __init__(self, batch_size):
    self._batch_size = batch_size
    self._cached = []

  def process(self, element):
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

  def process(self, element):
    for item in element:
      yield item


class LoadImagesDoFn(beam.DoFn):
  """A DoFn that reads image from url."""

  def process(self, element):
    from tensorflow.python.lib.io import file_io as tf_file_io

    with tf_file_io.FileIO(element['image_url'], 'r') as ff:
      image_bytes = ff.read()
    out_element = {'image_bytes': image_bytes}
    out_element.update(element)
    yield out_element


class PredictBatchDoFn(beam.DoFn):
  """A DoFn that does batch prediction."""

  def __init__(self, model_dir):
    self._model_dir = model_dir
    self._session = None
    self._tf_inputs = None
    self._tf_outputs = None

  def start_bundle(self, context=None):
    self._session, self._tf_inputs, self._tf_outputs = _load_tf_model(self._model_dir)

  def finish_bundle(self, context=None):
    if self._session is not None:
      self._session.close()

  def process(self, element):
    import collections

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
  """A DoFn that process prediction results by casting values and calculating
     target_prob.
  """

  def process(self, element):
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

  def process(self, element):
    import csv
    import StringIO

    line = StringIO.StringIO()
    if len(element) == 5:
      csv.DictWriter(line, ['image_url', 'target', 'predicted', 'target_prob',
                            'predicted_prob']).writerow(element)
    else:
      csv.DictWriter(line, ['image_url', 'predicted', 'predicted_prob']).writerow(element)
    yield line.getvalue()


def configure_pipeline(p, dataset, model_dir, output_csv, output_bq_table):
  """Configures a dataflow pipeline for batch prediction."""

  data = _util.get_sources_from_dataset(p, dataset, 'predict')
  if len(dataset.schema) == 2:
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
  results = (data |
             'Load Images' >> beam.ParDo(LoadImagesDoFn()) |
             'Batch Inputs' >> beam.ParDo(EmitAsBatchDoFn(20)) |
             'Batch Predict' >> beam.ParDo(PredictBatchDoFn(model_dir)) |
             'Unbatch' >> beam.ParDo(UnbatchDoFn()) |
             'Process Results' >> beam.ParDo(ProcessResultsDoFn()))

  if output_csv is not None:
    schema_file = output_csv + '.schema.json'
    results_save = (results |
                    'Prepare For Output' >> beam.ParDo(MakeCsvLineDoFn()) |
                    'Write Csv Results' >> beam.io.textio.WriteToText(output_csv,
                                                                      shard_name_template=''))
    (results_save |
     'Sample One' >> beam.transforms.combiners.Sample.FixedSizeGlobally(1) |
     'Serialize Schema' >> beam.Map(lambda path: json.dumps(output_schema)) |
     'Write Schema' >> beam.io.textio.WriteToText(schema_file, shard_name_template=''))

  if output_bq_table is not None:
    # BigQuery sink takes schema in the form of 'field1:type1,field2:type2...'
    bq_schema_string = ','.join(x['name'] + ':' + x['type'] for x in output_schema)
    sink = beam.io.BigQuerySink(output_bq_table, schema=bq_schema_string,
                                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE)
    results | 'Write BQ Results' >> beam.io.Write(sink)
