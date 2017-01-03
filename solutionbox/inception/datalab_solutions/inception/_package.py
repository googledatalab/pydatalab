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


"""Provides interface for Datalab. It provides:
     local_preprocess
     local_train
     local_predict
     cloud_preprocess
     cloud_train
     cloud_predict
   Datalab will look for functions with the above names.
"""

import logging
import os
import urllib

from . import _cloud
from . import _local
from . import _model
from . import _preprocess
from . import _trainer
from . import _util


def local_preprocess(input_csvs, labels_file, output_dir, checkpoint=None):
  print 'Local preprocessing...'
  # TODO: Move this to a new process to avoid pickling issues
  _local.Local(checkpoint).preprocess(input_csvs, labels_file, output_dir)
  print 'Done'


def cloud_preprocess(input_csvs, labels_file, output_dir, project, checkpoint=None,
                     pipeline_option=None):
  # TODO: Move this to a new process to avoid pickling issues
  _cloud.Cloud(project, checkpoint).preprocess(input_csvs, labels_file, output_dir,
      pipeline_option)
  if (_util.is_in_IPython()):
    import IPython
    dataflow_url = 'https://console.developers.google.com/dataflow?project=%s' % project
    html = 'Job submitted.'
    html += '<p>Click <a href="%s" target="_blank">here</a> to track preprocessing job. <br/>' \
        % dataflow_url
    IPython.display.display_html(html, raw=True)


def local_train(labels_file, input_dir, batch_size, max_steps, output_path, checkpoint=None):
  logger = logging.getLogger()
  original_level = logger.getEffectiveLevel()
  logger.setLevel(logging.INFO)
  print 'Local training...'
  try:
    _local.Local(checkpoint).train(labels_file, input_dir, batch_size, max_steps, output_path)
  finally:
    logger.setLevel(original_level)
  print 'Done'


def cloud_train(labels_file, input_dir, batch_size, max_steps, output_path,
                project, credentials, region, scale_tier='BASIC', checkpoint=None):
  job_info = _cloud.Cloud(project, checkpoint).train(labels_file, input_dir, batch_size,
      max_steps, output_path, credentials, region, scale_tier)
  if (_util.is_in_IPython()):
    import IPython
    log_url_query_strings = {
      'project': project,
      'resource': 'ml.googleapis.com/job_id/' + job_info['jobId']
    }
    log_url = 'https://console.developers.google.com/logs/viewer?' + \
        urllib.urlencode(log_url_query_strings)
    html = '<p>Click <a href="%s" target="_blank">here</a> to view cloud log. <br/>' % log_url
    IPython.display.display_html(html, raw=True)


def _display_predict_results(results, show_image):
  if (_util.is_in_IPython()):
    import IPython
    for image_file, label_and_score in results:
      if show_image is True:
        IPython.display.display_html('<p style="font-size:28px">%s(%.5f)</p>' % label_and_score,
            raw=True)
        IPython.display.display(IPython.display.Image(filename=image_file))
      else:
        IPython.display.display_html(
            '<p>%s&nbsp&nbsp%s(%.5f)</p>' % ((image_file,) + label_and_score), raw=True)
  else:
    print results


def local_predict(model_dir, image_files, labels_file, show_image=True):
  labels_and_scores = _local.Local().predict(model_dir, image_files, labels_file)
  results = zip(image_files, labels_and_scores)
  _display_predict_results(results, show_image)


def cloud_predict(model_id, image_files, labels_file, project, credentials, show_image=True):
  labels_and_scores = _cloud.Cloud(project).predict(model_id, image_files, labels_file,
                                                    credentials)
  results = zip(image_files, labels_and_scores)
  _display_predict_results(results, show_image)
