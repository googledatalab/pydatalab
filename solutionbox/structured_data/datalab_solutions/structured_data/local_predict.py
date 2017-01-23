# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for running predictions locally.

This module will always be run within a subprocess, and therefore normal
conventions of Cloud SDK do not apply here.
"""

from __future__ import print_function

import argparse
import json
import sys


def eprint(*args, **kwargs):
  """Print to stderr."""
  print(*args, file=sys.stderr, **kwargs)


VERIFY_TENSORFLOW_VERSION = ('Please verify the installed tensorflow version '
                             'with: "python -c \'import tensorflow; '
                             'print tensorflow.__version__\'".')

VERIFY_CLOUDML_VERSION = ('Please verify the installed cloudml sdk version with'
                          ': "python -c \'import google.cloud.ml as cloudml; '
                          'print cloudml.__version__\'".')


def _has_required_package():
  """Check whether required packages with correct version are installed.

  Returns:
    Whether the required packages are installed.
  """

  packages_ok = True

  # Check tensorflow with a recent version is installed.
  try:
    # pylint: disable=g-import-not-at-top
    import tensorflow as tf
    # pylint: enable=g-import-not-at-top
  except ImportError:
    eprint('Cannot import Tensorflow. Please verify '
           '"python -c \'import tensorflow\'" works.')
    packages_ok = False
  try:
    if tf.__version__ < '0.10.0':
      eprint('Tensorflow version must be at least 0.10.0. ',
             VERIFY_TENSORFLOW_VERSION)
      packages_ok = False
  except (NameError, AttributeError) as e:
    eprint('Error while getting the installed TensorFlow version: ', e,
           '\n', VERIFY_TENSORFLOW_VERSION)
    packages_ok = False

  # Check cloud ml sdk with a recent version is installed.
  try:
    # pylint: disable=g-import-not-at-top
    import google.cloud.ml as cloudml
    # pylint: enable=g-import-not-at-top
  except ImportError:
    eprint('Cannot import google.cloud.ml. Please verify '
           '"python -c \'import google.cloud.ml\'" works.')
    packages_ok = False
  try:
    if cloudml.__version__ < '0.1.7':
      eprint('Cloudml SDK version must be at least 0.1.7 '
             'to run local prediction. ', VERIFY_CLOUDML_VERSION)
      packages_ok = False
  except (NameError, AttributeError) as e:
    eprint('Error while getting the installed Cloudml SDK version: ', e,
           '\n', VERIFY_CLOUDML_VERSION)
    packages_ok = False

  return packages_ok


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-dir', required=True, help='Path of the model.')
  args, _ = parser.parse_known_args()
  if not _has_required_package():
    sys.exit(-1)

  instances = []
  for line in sys.stdin:
    instance = json.loads(line.rstrip('\n'))
    instances.append(instance)

  # pylint: disable=g-import-not-at-top
  from google.cloud.ml import prediction
  # pylint: enable=g-import-not-at-top
  print('gong to call prediction with ')
  print(instances)
  predictions = prediction.local_predict(model_dir=args.model_dir,
                                         instances=instances)
  print(json.dumps(predictions))


if __name__ == '__main__':
  main()
