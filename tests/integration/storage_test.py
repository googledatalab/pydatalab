"""Integration tests for google.datalab.storage."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import random
import string
import unittest

import google.datalab
from google.datalab import storage


class StorageTest(unittest.TestCase):

  def setUp(self):
    self._context = google.datalab.Context.default()
    logging.info('Using project: %s', self._context.project_id)

    suffix = ''.join(random.choice(string.lowercase) for _ in range(8))
    self._test_bucket_name = '{}-{}'.format(self._context.project_id, suffix)
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
