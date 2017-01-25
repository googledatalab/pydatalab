# Copyright 2015 Google Inc. All rights reserved.
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

from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import str
import mock
from oauth2client.client import AccessTokenCredentials
import unittest

import google.datalab.bigquery
import google.datalab.context
import google.datalab.utils


class TestCases(unittest.TestCase):

  def _check_name_parts(self, dataset):
    parsed_name = dataset._name_parts
    self.assertEqual('test', parsed_name[0])
    self.assertEqual('requestlogs', parsed_name[1])
    self.assertEqual('test:requestlogs', dataset._full_name)
    self.assertEqual('test:requestlogs', str(dataset))

  def test_parse_full_name(self):
    dataset = TestCases._create_dataset('test:requestlogs')
    self._check_name_parts(dataset)

  def test_parse_local_name(self):
    dataset = TestCases._create_dataset('requestlogs')
    self._check_name_parts(dataset)

  def test_parse_dict_full_name(self):
    dataset = TestCases._create_dataset({'project_id': 'test', 'dataset_id': 'requestlogs'})
    self._check_name_parts(dataset)

  def test_parse_dict_local_name(self):
    dataset = TestCases._create_dataset({'dataset_id': 'requestlogs'})
    self._check_name_parts(dataset)

  def test_parse_named_tuple_name(self):
    dataset = TestCases._create_dataset(google.datalab.bigquery._utils.DatasetName('test', 'requestlogs'))
    self._check_name_parts(dataset)

  def test_parse_tuple_full_name(self):
    dataset = TestCases._create_dataset(('test', 'requestlogs'))
    self._check_name_parts(dataset)

  def test_parse_tuple_local(self):
    dataset = TestCases._create_dataset(('requestlogs'))
    self._check_name_parts(dataset)

  def test_parse_array_full_name(self):
    dataset = TestCases._create_dataset(['test', 'requestlogs'])
    self._check_name_parts(dataset)

  def test_parse_array_local(self):
    dataset = TestCases._create_dataset(['requestlogs'])
    self._check_name_parts(dataset)

  def test_parse_invalid_name(self):
    with self.assertRaises(Exception):
      _ = TestCases._create_dataset('today@')

  @mock.patch('google.datalab.bigquery._api.Api.datasets_get')
  def test_dataset_exists(self, mock_api_datasets_get):
    mock_api_datasets_get.return_value = ''
    dataset = TestCases._create_dataset('test:requestlogs')
    self.assertTrue(dataset.exists())
    mock_api_datasets_get.side_effect = google.datalab.utils.RequestException(404, None)
    dataset._info = None
    self.assertFalse(dataset.exists())

  @mock.patch('google.datalab.bigquery._api.Api.datasets_insert')
  @mock.patch('google.datalab.bigquery._api.Api.datasets_get')
  def test_datasets_create_fails(self, mock_api_datasets_get, mock_api_datasets_insert):
    mock_api_datasets_get.side_effect = google.datalab.utils.RequestException(None, 404)
    mock_api_datasets_insert.return_value = {}

    ds = TestCases._create_dataset('requestlogs')
    with self.assertRaises(Exception):
      _ = ds.create()

  @mock.patch('google.datalab.bigquery._api.Api.datasets_insert')
  @mock.patch('google.datalab.bigquery._api.Api.datasets_get')
  def test_datasets_create_succeeds(self, mock_api_datasets_get, mock_api_datasets_insert):
    mock_api_datasets_get.side_effect = google.datalab.utils.RequestException(404, None)
    mock_api_datasets_insert.return_value = {'selfLink': None}
    ds = TestCases._create_dataset('requestlogs')
    self.assertEqual(ds, ds.create())

  @mock.patch('google.datalab.bigquery._api.Api.datasets_insert')
  @mock.patch('google.datalab.bigquery._api.Api.datasets_get')
  def test_datasets_create_redundant(self, mock_api_datasets_get, mock_api_datasets_insert):
    ds = TestCases._create_dataset('requestlogs', {})
    mock_api_datasets_get.return_value = None
    mock_api_datasets_insert.return_value = {}
    self.assertEqual(ds, ds.create())

  @mock.patch('google.datalab.bigquery._api.Api.datasets_get')
  @mock.patch('google.datalab.bigquery._api.Api.datasets_delete')
  def test_datasets_delete_succeeds(self, mock_api_datasets_delete, mock_api_datasets_get):
    mock_api_datasets_get.return_value = ''
    mock_api_datasets_delete.return_value = None
    ds = TestCases._create_dataset('requestlogs')
    self.assertIsNone(ds.delete())

  @mock.patch('google.datalab.bigquery._api.Api.datasets_get')
  @mock.patch('google.datalab.bigquery._api.Api.datasets_delete')
  def test_datasets_delete_fails(self, mock_api_datasets_delete, mock_api_datasets_get):
    mock_api_datasets_delete.return_value = None
    mock_api_datasets_get.side_effect = google.datalab.utils.RequestException(404, None)
    ds = TestCases._create_dataset('requestlogs')
    with self.assertRaises(Exception):
      _ = ds.delete()

  @mock.patch('google.datalab.bigquery._api.Api.tables_list')
  def test_tables_list(self, mock_api_tables_list):
    mock_api_tables_list.return_value = {
      'tables': [
          {
            'type': 'TABLE',
            'tableReference': {'projectId': 'p', 'datasetId': 'd', 'tableId': 't1'}
          },
          {
            'type': 'TABLE',
            'tableReference': {'projectId': 'p', 'datasetId': 'd', 'tableId': 't2'}
          },
      ]
    }
    ds = TestCases._create_dataset('requestlogs')
    tables = [table for table in ds]
    self.assertEqual(2, len(tables))
    self.assertEqual('p:d.t1', str(tables[0]))
    self.assertEqual('p:d.t2', str(tables[1]))

  @mock.patch('google.datalab.bigquery.Dataset._get_info')
  @mock.patch('google.datalab.bigquery._api.Api.datasets_list')
  def test_datasets_list(self, mock_api_datasets_list, mock_dataset_get_info):
    mock_api_datasets_list.return_value = {
      'datasets': [
        {'datasetReference': {'projectId': 'p', 'datasetId': 'd1'}},
        {'datasetReference': {'projectId': 'p', 'datasetId': 'd2'}},
      ]
    }
    mock_dataset_get_info.return_value = {}
    datasets = [dataset for dataset in google.datalab.bigquery.Datasets('test',
                                                                 TestCases._create_context())]
    self.assertEqual(2, len(datasets))
    self.assertEqual('p:d1', str(datasets[0]))
    self.assertEqual('p:d2', str(datasets[1]))

  @mock.patch('google.datalab.bigquery._api.Api.tables_list')
  @mock.patch('google.datalab.bigquery._api.Api.datasets_get')
  @mock.patch('google.datalab.bigquery._api.Api.datasets_update')
  def test_datasets_update(self, mock_api_datasets_update, mock_api_datasets_get,
                           mock_api_tables_list):
    mock_api_tables_list.return_value = {
      'tables': [
        {'type': 'TABLE', 'tableReference': {'projectId': 'p', 'datasetId': 'd', 'tableId': 't1'}},
        {'type': 'TABLE', 'tableReference': {'projectId': 'p', 'datasetId': 'd', 'tableId': 't2'}},
      ]
    }
    info = {'friendlyName': 'casper', 'description': 'ghostly logs'}
    mock_api_datasets_get.return_value = info
    ds = TestCases._create_dataset('requestlogs')

    new_friendly_name = 'aziraphale'
    new_description = 'demon duties'
    ds.update(new_friendly_name, new_description)

    name, info = mock_api_datasets_update.call_args[0]
    self.assertEqual(ds.name, name)

    self.assertEqual(new_friendly_name, ds.friendly_name)
    self.assertEqual(new_description, ds.description)

  @staticmethod
  def _create_context():
    project_id = 'test'
    creds = AccessTokenCredentials('test_token', 'test_ua')
    return google.datalab.context.Context(project_id, creds)

  @staticmethod
  def _create_dataset(name, metadata=None):
    # Patch get_info so we don't have to mock it everywhere else.
    orig = google.datalab.bigquery.Dataset._get_info
    google.datalab.bigquery.Dataset._get_info = mock.Mock(return_value=metadata)
    ds =  google.datalab.bigquery.Dataset(name, context=TestCases._create_context())
    google.datalab.bigquery.Dataset._get_info = orig
    return ds
