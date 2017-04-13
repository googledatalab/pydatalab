"""Integration tests for google.datalab.storage."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import random
import string
import subprocess
import unittest

import google.datalab
from google.datalab import storage


class StorageTest(unittest.TestCase):

  def setUp(self):
    project_id = os.environ.get('PROJECT_ID', '')
    if not project_id:
      # If there's no project ID set, we try asking gcloud. We need to send
      # stderr to /dev/null, otherwise we get some spurious output.
      gcloud_output = subprocess.check_output([
          'gcloud', 'config', 'get-value', 'project'], stderr=subprocess.PIPE)
      project_id = gcloud_output.strip()
    logging.info('Using project ID: %s', project_id)

    context = google.datalab.Context.default()
    context.set_project_id(project_id)
    self._context = context

    suffix = ''.join(random.choice(string.letters) for _ in range(8))
    self._test_bucket_name = '-'.join((project_id, suffix)).lower()
    logging.info('test bucket: %s', self._test_bucket_name)

  def test_object_deletion_consistency(self):
    b = storage.Bucket(self._test_bucket_name, context=self._context)
    b.create()
    o = b.object('sample')
    o.write_stream('contents', 'text/plain')
    o.delete()
    b.delete()


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  unittest.main()
