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

"""Implements Cloud ML Analysis Helpers"""


import google.cloud.ml as ml
import numpy as np
import pandas as pd
import yaml

import datalab.bigquery as bq


def csv_to_dataframe(csv_path, schema_path):
  """Given a CSV file together with its BigQuery schema file in yaml, load
     content into a dataframe.

    Args:
      csv_path: Input CSV path. Can be local or GCS.
      schema_path: Input schema path. Can be local or GCS.

    Returns:
      Loaded pandas dataframe.
  """
  with ml.util._file.open_local_or_gcs(schema_path, mode='r') as f:
    schema = yaml.safe_load(f)
  _MAPPINGS = {
    'FLOAT': np.float64,
    'INTEGER': np.int64,
    'TIMESTAMP': np.datetime64,
    'BOOLEAN': np.bool,
  }
  for item in schema:
    item['type'] = _MAPPINGS.get(item['type'], object)
  names = [x['name'] for x in schema]
  dtype = {x['name']: x['type'] for x in schema}
  with ml.util._file.open_local_or_gcs(csv_path, mode='r') as f:
    return pd.read_csv(f, names=names, dtype=dtype)
