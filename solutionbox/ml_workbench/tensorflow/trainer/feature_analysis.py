# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import csv
import json
import os
import pandas as pd
import sys
import six
from tensorflow.python.lib.io import file_io

from . import feature_transforms as constant


def check_schema_transforms_match(schema, inverted_features):
  """Checks that the transform and schema do not conflict.

  Args:
    schema: schema list
    inverted_features: inverted_features dict

  Raises:
    ValueError if transform cannot be applied given schema type.
  """
  num_target_transforms = 0

  for col_schema in schema:
    col_name = col_schema['name']
    col_type = col_schema['type'].lower()

    # Check each transform and schema are compatible
    if col_name in inverted_features:
      for transform in inverted_features[col_name]:
        transform_name = transform['transform']
        if transform_name == constant.TARGET_TRANSFORM:
          num_target_transforms += 1
          continue

        elif col_type in constant.NUMERIC_SCHEMA:
          if transform_name not in constant.NUMERIC_TRANSFORMS:
            raise ValueError(
                'Transform %s not supported by schema %s' % (transform_name, col_type))
        elif col_type == constant.STRING_SCHEMA:
          if (transform_name not in constant.CATEGORICAL_TRANSFORMS + constant.TEXT_TRANSFORMS and
             transform_name != constant.IMAGE_TRANSFORM):
            raise ValueError(
                'Transform %s not supported by schema %s' % (transform_name, col_type))
        else:
          raise ValueError('Unsupported schema type %s' % col_type)

    # Check each transform is compatible for the same source column.
    # inverted_features[col_name] should belong to exactly 1 of the 5 groups.
    if col_name in inverted_features:
      transform_set = {x['transform'] for x in inverted_features[col_name]}
      if 1 != sum([transform_set.issubset(set(constant.NUMERIC_TRANSFORMS)),
                   transform_set.issubset(set(constant.CATEGORICAL_TRANSFORMS)),
                   transform_set.issubset(set(constant.TEXT_TRANSFORMS)),
                   transform_set.issubset(set([constant.IMAGE_TRANSFORM])),
                   transform_set.issubset(set([constant.TARGET_TRANSFORM]))]):
        message = """
          The source column of a feature can only be used in multiple
          features within the same family of transforms. The familes are

          1) text transformations: %s
          2) categorical transformations: %s
          3) numerical transformations: %s
          4) image transformations: %s
          5) target transform: %s

          Any column can also be a key column.

          But column %s is used by transforms %s.
          """ % (str(constant.TEXT_TRANSFORMS),
                 str(constant.CATEGORICAL_TRANSFORMS),
                 str(constant.NUMERIC_TRANSFORMS),
                 constant.IMAGE_TRANSFORM,
                 constant.TARGET_TRANSFORM,
                 col_name,
                 str(transform_set))
        raise ValueError(message)

  if num_target_transforms != 1:
    raise ValueError('Must have exactly one target transform')


def save_schema_features(schema, features, output):
  # Save a copy of the schema and features in the output folder.
  file_io.write_string_to_file(
    os.path.join(output, constant.SCHEMA_FILE),
    json.dumps(schema, indent=2))

  file_io.write_string_to_file(
    os.path.join(output, constant.FEATURES_FILE),
    json.dumps(features, indent=2))


def expand_defaults(schema, features):
  """Add to features any default transformations.

  Not every column in the schema has an explicit feature transformation listed
  in the featurs file. For these columns, add a default transformation based on
  the schema's type. The features dict is modified by this function call.

  After this function call, every column in schema is used in a feature, and
  every feature uses a column in the schema.

  Args:
    schema: schema list
    features: features dict

  Raises:
    ValueError: if transform cannot be applied given schema type.
  """

  schema_names = [x['name'] for x in schema]

  # Add missing source columns
  for name, transform in six.iteritems(features):
    if 'source_column' not in transform:
      transform['source_column'] = name

  # Check source columns are in the schema and collect which are used.
  used_schema_columns = []
  for name, transform in six.iteritems(features):
    if transform['source_column'] not in schema_names:
      raise ValueError('source column %s is not in the schema for transform %s'
                       % (transform['source_column'], name))
    used_schema_columns.append(transform['source_column'])

  # Update default transformation based on schema.
  for col_schema in schema:
    schema_name = col_schema['name']
    schema_type = col_schema['type'].lower()

    if schema_type not in constant.NUMERIC_SCHEMA + [constant.STRING_SCHEMA]:
      raise ValueError(('Only the following schema types are supported: %s'
                        % ' '.join(constant.NUMERIC_SCHEMA + [constant.STRING_SCHEMA])))

    if schema_name not in used_schema_columns:
      # add the default transform to the features
      if schema_type in constant.NUMERIC_SCHEMA:
        features[schema_name] = {
            'transform': constant.DEFAULT_NUMERIC_TRANSFORM,
            'source_column': schema_name}
      elif schema_type == constant.STRING_SCHEMA:
        features[schema_name] = {
            'transform': constant.DEFAULT_CATEGORICAL_TRANSFORM,
            'source_column': schema_name}
      else:
        raise NotImplementedError('Unknown type %s' % schema_type)


# TODO(brandondutra): introduce the notion an analysis plan/classes if we
# support more complicated transforms like binning by quratiles.
def invert_features(features):
  """Make a dict in the form source column : set of transforms.

  Note that the key transform is removed.
  """
  inverted_features = collections.defaultdict(list)
  for transform in six.itervalues(features):
    source_column = transform['source_column']
    if transform['transform'] == constant.KEY_TRANSFORM:
      continue
    inverted_features[source_column].append(transform)

  return dict(inverted_features)  # convert from defaultdict to dict


def run_local_analysis(output_dir, csv_file_pattern, schema, features):
  """Use pandas to analyze csv files.

  Produces a stats file and vocab files.

  Args:
    output_dir: output folder
    csv_file_pattern: list of csv file paths, may contain wildcards
    schema: CSV schema list
    features: features config

  Raises:
    ValueError: on unknown transfrorms/schemas
  """
  sys.stdout.write('Expanding any file patterns...\n')
  sys.stdout.flush()
  header = [column['name'] for column in schema]
  input_files = []
  for file_pattern in csv_file_pattern:
    input_files.extend(file_io.get_matching_files(file_pattern))
  sys.stdout.write('file list computed.\n')
  sys.stdout.flush()

  expand_defaults(schema, features)  # features are updated.
  inverted_features = invert_features(features)
  check_schema_transforms_match(schema, inverted_features)

  # Make a copy of inverted_features and update the target transform to be
  # identity or one hot depending on the schema.
  inverted_features_target = copy.deepcopy(inverted_features)
  for name, transforms in six.iteritems(inverted_features_target):
    transform_set = {x['transform'] for x in transforms}
    if transform_set == set([constant.TARGET_TRANSFORM]):
      target_schema = next(col['type'].lower() for col in schema if col['name'] == name)
      if target_schema in constant.NUMERIC_SCHEMA:
        inverted_features_target[name] = [{'transform': constant.IDENTITY_TRANSFORM}]
      else:
        inverted_features_target[name] = [{'transform': constant.ONE_HOT_TRANSFORM}]

  # initialize the results
  def _init_numerical_results():
    return {'min': float('inf'),
            'max': float('-inf'),
            'count': 0,
            'sum': 0.0}
  numerical_results = collections.defaultdict(_init_numerical_results)
  vocabs = collections.defaultdict(lambda: collections.defaultdict(int))

  num_examples = 0
  # for each file, update the numerical stats from that file, and update the set
  # of unique labels.
  for input_file in input_files:
    sys.stdout.write('Analyzing file %s...\n' % input_file)
    sys.stdout.flush()
    with file_io.FileIO(input_file, 'r') as f:
      for line in csv.reader(f):
        if len(header) != len(line):
          raise ValueError('Schema has %d columns but a csv line only has %d columns.' %
                           (len(header), len(line)))
        parsed_line = dict(zip(header, line))
        num_examples += 1

        for col_name, transform_set in six.iteritems(inverted_features_target):
          # All transforms in transform_set require the same analysis. So look
          # at the first transform.
          transform = next(iter(transform_set))
          if transform['transform'] in constant.TEXT_TRANSFORMS:
            separator = transform.get('separator', ' ')
            split_strings = parsed_line[col_name].split(separator)

            # If a label is in the row N times, increase it's vocab count by 1.
            # This is needed for TFIDF, but it's also an interesting stat.
            for one_label in set(split_strings):
              # Filter out empty strings
              if one_label:
                vocabs[col_name][one_label] += 1
          elif transform['transform'] in constant.CATEGORICAL_TRANSFORMS:
            if parsed_line[col_name]:
              vocabs[col_name][parsed_line[col_name]] += 1
          elif transform['transform'] in constant.NUMERIC_TRANSFORMS:
            if not parsed_line[col_name].strip():
              continue

            numerical_results[col_name]['min'] = (
              min(numerical_results[col_name]['min'],
                  float(parsed_line[col_name])))
            numerical_results[col_name]['max'] = (
              max(numerical_results[col_name]['max'],
                  float(parsed_line[col_name])))
            numerical_results[col_name]['count'] += 1
            numerical_results[col_name]['sum'] += float(parsed_line[col_name])

    sys.stdout.write('file %s analyzed.\n' % input_file)
    sys.stdout.flush()

  # Write the vocab files. Each label is on its own line.
  vocab_sizes = {}
  for name, label_count in six.iteritems(vocabs):
    # df is now:
    # label1,count
    # label2,count
    # ...
    # where label1 is the most frequent label, and label2 is the 2nd most, etc.
    df = pd.DataFrame([{'label': label, 'count': count}
                       for label, count in sorted(six.iteritems(label_count),
                                                  key=lambda x: x[1],
                                                  reverse=True)],
                      columns=['label', 'count'])
    csv_string = df.to_csv(index=False, header=False)

    file_io.write_string_to_file(
        os.path.join(output_dir, constant.VOCAB_ANALYSIS_FILE % name),
        csv_string)

    vocab_sizes[name] = {'vocab_size': len(label_count)}

  # Update numerical_results to just have min/min/mean
  for col_name in numerical_results:
    if float(numerical_results[col_name]['count']) == 0:
      raise ValueError('Column %s has a zero count' % col_name)
    mean = (numerical_results[col_name]['sum'] /
            float(numerical_results[col_name]['count']))
    del numerical_results[col_name]['sum']
    del numerical_results[col_name]['count']
    numerical_results[col_name]['mean'] = mean

  # Write the stats file.
  numerical_results.update(vocab_sizes)
  stats = {'column_stats': numerical_results, 'num_examples': num_examples}
  file_io.write_string_to_file(
      os.path.join(output_dir, constant.STATS_FILE),
      json.dumps(stats, indent=2, separators=(',', ': ')))

  save_schema_features(schema, features, output_dir)
