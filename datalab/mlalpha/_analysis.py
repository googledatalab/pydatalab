# Copyright 2016 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Implements Cloud ML Eval Results Analysis"""

import apache_beam as beam
from collections import namedtuple

"""Prepresents an eval results CSV file. For example, the content is like:
     107,Iris-versicolor,1.64827824278e-07,0.999999880791,6.27104979056e-10
     100,Iris-versicolor,3.5338824091e-05,0.99996471405,1.32811195375e-09
     ...
"""
CsvEvalResults = namedtuple('CsvEvalResults', 'source, key_index predicted_index score_index_start num_scores')

"""Prepresents an eval source CSV file. For example, the content is like:
     107,Iris-virginica,4.9,2.5,4.5,1.7
     100,Iris-versicolor,5.7,2.8,4.1,1.3
     ...
   The metadata is generated in the preprocessing pipeline. It is used to describe the CSV file,
   including schema, headers, etc.
"""
CsvEvalSource = namedtuple('CsvEvalSource', 'source metadata')


class EvalResultsCsvCoder(beam.coders.Coder):
  """A coder to read from Eval results CSV file. Note encode() is only needed in cloud run.
  """
  def __init__(self, eval_results):
    self._eval_results = eval_results

  def decode(self, csv_line):
    import csv
    source_elem = next(csv.reader([csv_line]))
    key = source_elem[self._eval_results.key_index]
    element = {
        'predicted': source_elem[self._eval_results.predicted_index],
        'scores': source_elem[self._eval_results.score_index_start: \
            self._eval_results.score_index_start+self._eval_results.num_scores]
    }
    return (key, element)

  def encode(self, element):
    return str(element)


class AccuracyFn(beam.CombineFn):
  """A transform to compute accuracy for feature slices.
  """
  def __init__(self, target_column_name):
    self._target_column_name = target_column_name

  def create_accumulator(self):
    return (0.0, 0)

  def add_input(self, (sum, count), input):
    new_sum = sum
    if (input['predicted'] == input[self._target_column_name]):
      new_sum += 1
    return new_sum, count + 1

  def merge_accumulators(self, accumulators):
    sums, counts = zip(*accumulators)
    return sum(sums), sum(counts)

  def extract_output(self, (sum, count)):
    accuracy = float(sum) / count if count else float('NaN')
    return {'accuracy': accuracy, 'totalWeightedExamples': count}


class FeatureSlicingPipeline(object):
  """The pipeline to generate feature slicing stats. For example, accuracy values given 
     "species = Iris-versicolor", "education = graduate", etc.
     It is implemented with DataFlow.
  """
  @staticmethod
  def _pair_source_with_key(element):
    key = element['key']
    del element['key']
    return (key, element)

  @staticmethod
  def _join_info((key, info)):
    value = info['source'][0]
    value.update(info['results'][0])
    return (key, value)

  def _pipeline_def(self, p, eval_source, eval_results, features_to_slice, metrics, output_file,
                    shard_name_template=None):
    import datalab.mlalpha as mlalpha
    import google.cloud.ml.io as io
    import json

    metadata = mlalpha.Metadata(eval_source.metadata)
    target_name, _ = metadata.get_target_name_and_scenario()

    # Load eval source.
    eval_source_coder = io.CsvCoder(metadata.get_csv_headers(), metadata.get_numeric_columns())
    eval_source_data = p | beam.io.ReadFromText(eval_source.source, coder=eval_source_coder) | \
        beam.Map('pair_source_with_key', FeatureSlicingPipeline._pair_source_with_key)

    # Load eval results.
    eval_results_data = p | \
        beam.Read('ReadEvalResults', beam.io.TextFileSource(eval_results.source,
            coder=EvalResultsCsvCoder(eval_results)))

    # Join source with results by key.
    joined_results = {'source': eval_source_data, 'results': eval_results_data} | \
        beam.CoGroupByKey() | beam.Map('join by key', FeatureSlicingPipeline._join_info)

    feature_metrics_list = []
    for feature_to_slice in features_to_slice:
      feature_metrics = joined_results | \
          beam.Map('slice_get_key_%s' % feature_to_slice,
                   lambda (k,v),f=feature_to_slice: (v[f], v)) | \
          beam.CombinePerKey('slice_combine_%s' % feature_to_slice,
                             AccuracyFn(target_name)) | \
          beam.Map('slice_prepend_feature_name_%s' % feature_to_slice,
                   lambda (k,v),f=feature_to_slice: ('%s:%s' % (f, k), v))
      feature_metrics_list.append(feature_metrics)

    feature_metrics_list | beam.Flatten() | \
        beam.Map('ToJsonFormat', lambda (k,v): json.dumps({'feature': k, 'metricValues': v})) | \
        beam.io.WriteToText(output_file, shard_name_template=shard_name_template)
    return p


  def run_local(self, eval_source, eval_results, features_to_slice, metrics, output_file):
    """Run the pipeline locally. Blocks execution until it finishes.

    Args:
      eval_source: The only supported format is CsvEvalResults now while we may add more.
                   Note the source can be either a GCS path or a local path.
      eval_results: The only supported format is CsvEvalSource now while we may add more.
                    Note the source can be either a GCS path or a local path.
      features_to_slice: A list of features to slice on. The features must exist in
                         eval_source, and can be numeric, categorical, or target.
      metrics: A list of metrics to compute. For classification, it supports "accuracy",
               "logloss". For regression, it supports "RMSE".
      output_file: The path to a local file holding the aggregated results.
    """
    p = beam.Pipeline('DirectPipelineRunner')
    self._pipeline_def(p, eval_source, eval_results, features_to_slice, metrics, output_file,
                       shard_name_template='')
    p.run()


  def default_pipeline_options(self, output_dir):
    """Get default DataFlow options. Users can customize it further on top of it and then
       send the option to run_cloud().

    Args:
      output_dir: A GCS path which will be used as base path for tmp and staging dir.

    Returns:
      A dictionary of options.
    """
    import datalab.context as context
    import datetime
    import google.cloud.ml as ml
    import os

    options = {
        'staging_location': os.path.join(output_dir, 'tmp', 'staging'),
        'temp_location': os.path.join(output_dir, 'tmp'),
        'job_name': 'feature-slicing-pipeline' + '-' + \
             datetime.datetime.now().strftime('%y%m%d-%H%M%S'),
        'project': context.Context.default().project_id,
        'extra_packages': ['gs://cloud-datalab/dataflow/datalab.tar.gz', ml.sdk_location],
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'no_save_main_session': True
      }
    return options
  
  def run_cloud(self, eval_source, eval_results, features_to_slice, metrics, output_file,
                pipeline_option=None):
    """Run the pipeline in cloud. Returns when the job is submitted.
       Calling of this function may incur some cost since it runs a DataFlow job in Google Cloud.
       If pipeline_option is not specified, make sure you are signed in (through Datalab)
       and a default project is set so it can get credentials and projects from global context.

    Args:
      eval_source: The only supported format is CsvEvalResults now while we may add more.
                   The source needs to be a GCS path and is readable to current signed in user.
      eval_results: The only supported format is CsvEvalSource now while we may add more.
                    The source needs to be a GCS path and is readable to current signed in user.
      features_to_slice: A list of features to slice on. The features must exist in
                         eval_source, and can be numeric, categorical, or target.
      metrics: A list of metrics to compute. For classification, it supports "accuracy",
               "logloss". For regression, it supports "RMSE".
      pipeline_option: If not specified, use default options. Recommend customizing your options
                       based on default one obtained from default_pipeline_options(). For example,
                         options = fsp.default_pipeline_options()
                         options['num_workers'] = 10
                         ...
      output_file: A GCS file prefix holding the aggregated results.
    """
    import os
    if pipeline_option is None:
      output_dir = os.path.dirname(output_file)
      pipeline_option = self.default_pipeline_options(output_dir)
    opts = beam.pipeline.PipelineOptions(flags=[], **pipeline_option)
    p = beam.Pipeline('DataflowPipelineRunner', options=opts)
    self._pipeline_def(p, eval_source, eval_results, features_to_slice, metrics, output_file)
    p.run()

