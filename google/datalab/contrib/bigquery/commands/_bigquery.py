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

"""Google Cloud Platform library - BigQuery IPython Functionality."""
from builtins import str
import google
import google.datalab.utils as utils


def _create_pipeline_subparser(parser):
  pipeline_parser = parser.subcommand('pipeline', 'Creates a pipeline to execute a SQL query to '
                                                  'transform data using BigQuery.')

  # common arguments
  pipeline_parser.add_argument('-b', '--billing', type=int, help='BigQuery billing tier',
                               required=True)
  pipeline_parser.add_argument('-n', '--name', type=str, help='BigQuery pipeline name',
                               required=True)

  return pipeline_parser


def _pipeline_cell(args, cell_body):
    """Implements the pipeline subcommand in the %%bq magic.

    The supported syntax is:

        %%bq pipeline <args>
        [<inline YAML>]
        TODO(rajivpb): Add schema here so that it's clear from the documentation what the expected
        format is. https://github.com/googledatalab/pydatalab/issues/499.
    Args:
      args: the arguments following '%%bq pipeline'.
      cell_body: the contents of the cell
    """
    name = args.get('name')
    if name is None:
        raise Exception("Pipeline name was not specified.")

    bq_pipeline_config = utils.commands.parse_config(
        cell_body, utils.commands.notebook_environment())
    pipeline_spec = _get_pipeline_spec_from_config(bq_pipeline_config)
    pipeline = google.datalab.contrib.pipeline._pipeline.Pipeline(name, pipeline_spec)
    utils.commands.notebook_environment()[name] = pipeline

    # TODO(rajivpb): See https://github.com/googledatalab/pydatalab/issues/501. Don't return python.
    return pipeline.get_airflow_spec


def _get_pipeline_spec_from_config(bq_pipeline_config):
  input_config = bq_pipeline_config.get('input', None)
  transformation_config = bq_pipeline_config.get('transformation', None)
  output_config = bq_pipeline_config.get('output', None)

  load_task_config_name = 'bq_pipeline_load_task'
  execute_task_config_name = 'bq_pipeline_execute_task'
  extract_task_config_name = 'bq_pipeline_extract_task'

  load_task_config = _get_load_parameters(input_config)
  execute_task_config = _get_execute_parameters(load_task_config_name, transformation_config)
  extract_task_config = _get_extract_parameters(execute_task_config_name, execute_task_config,
                                                output_config)
  pipeline_spec = {
    'schedule': bq_pipeline_config['schedule'],
  }

  pipeline_spec['tasks'] = {}
  if load_task_config:
    pipeline_spec['tasks'][load_task_config_name] = load_task_config
  if execute_task_config:
    pipeline_spec['tasks'][execute_task_config_name] = execute_task_config
  if extract_task_config:
    pipeline_spec['tasks'][extract_task_config_name] = extract_task_config

  if not load_task_config and not execute_task_config and not extract_task_config:
    raise Exception('Pipeline has no tasks to execute.')

  return pipeline_spec


def _get_load_parameters(bq_pipeline_input_config):
    load_task_config = {'type': 'pydatalab.bq.load'}
    path_exists = False
    if 'path' in bq_pipeline_input_config:
      # The path URL of the GCS load file(s).
      load_task_config['path'] = bq_pipeline_input_config['path']
      path_exists = True

    table_exists = False
    if 'table' in bq_pipeline_input_config:
      # The destination bigquery table name for loading
      load_task_config['table'] = bq_pipeline_input_config['table']
      table_exists = True

    schema_exists = False
    if 'schema' in bq_pipeline_input_config:
      # The schema of the destination bigquery table
      load_task_config['schema'] = bq_pipeline_input_config['schema']
      schema_exists = True

    # We now figure out whether a load operation is required
    if table_exists:
      if path_exists:
        if schema_exists:
          # One of 'create' (default), 'append' or 'overwrite' for loading data into BigQuery. If a
          # schema is specified, we assume that the table needs to be created.
          load_task_config['mode'] = 'create'
        else:
          # If a schema is not specified, we assume that the table needs to be appended, since this
          # is the most likely scenario for users running pipelines.
          # TODO(rajivpb): Is the above assumption reasonable?
          load_task_config['mode'] = 'append'
      else:
        # If table exists, but a path does not, then we have our data in BQ already and no load is
        # required.
        return None
    else:
      # If the table doesn't exist, but a path does, then it's likely an extended data-source (and
      # the schema would need to be either present or auto-detected).
      if not path_exists:
        if 'format' in bq_pipeline_input_config:  # Some parameter validation
          raise Exception('Path is not specified, but a format is.')
        # If neither table or path exist, there is no load to be done.
        return None

    assert(path_exists == True)

    # One of 'csv' (default) or 'json' for the format of the load file.
    load_task_config['format'] = bq_pipeline_input_config.get('format', 'csv')
    if load_task_config['format'] == 'csv':
      csv_config = bq_pipeline_input_config.get('csv', {})
      # The inter-field delimiter for CVS (default ,) in the load file
      load_task_config['delimiter'] = csv_config.get('delimiter', ',')
      # The quoted field delimiter for CVS (default ") in the load file
      load_task_config['quote'] = csv_config.get('quote', '"')
      # The number of head lines (default is 0) to skip during load; useful for CSV
      load_task_config['skip'] = csv_config.get('skip', 0)
      # Reject bad values and jagged lines when loading (default True)
      load_task_config['strict'] = csv_config.get('strict', True)

    return load_task_config


def _get_execute_parameters(load_task_config_name, bq_pipeline_transformation_config):
    execute_task_config = {
      'type': 'pydatalab.bq.execute',
      'up_stream': [load_task_config_name]
    }

    # The name of query for execution; if absent, we return None as we assume that there is
    # no query to execute
    if 'query' not in bq_pipeline_transformation_config:
      if any(key in bq_pipeline_transformation_config for key in ['large', 'mode']):
        raise Exception('Query is not specified, but at least one query option is.')
      return None

    execute_task_config['query'] = bq_pipeline_transformation_config['query']

    # TODO(rajivpb): There is no unit-test coverage of this.

    # One of 'create' (default), 'append' or 'overwrite' for the destination table in BigQuery
    execute_task_config['mode'] = bq_pipeline_transformation_config.get('mode', 'create')

    return execute_task_config


def _get_extract_parameters(execute_task_config_name, execute_task_config,
                            bq_pipeline_output_config):
    extract_task_config = {
      'type': 'pydatalab.bq.extract',
      'up_stream': [execute_task_config_name]
    }

    # Destination table name for the execution results. When present, this will need to be set in
    # execute_task_config. When absent, it means that there is no extraction to be done, so we
    # return None.
    if 'table' not in bq_pipeline_output_config:
      return None

    execute_task_config['table'] = bq_pipeline_output_config['table']
    extract_task_config['table'] = bq_pipeline_output_config['table']

    extract_task_config['path'] = bq_pipeline_output_config.get('path')
    if not extract_task_config['path']:
      # If a path is not specified, there is nothing to extract, so we return None after making
      # sure that format is not specified.
      if 'format' in bq_pipeline_output_config:
        raise Exception('Path is not specified, but format is.')
      return None

    # One of 'csv' (default) or 'json' for the format of the load file.
    extract_task_config['format'] = bq_pipeline_output_config.get('format', 'csv')
    if extract_task_config['format'] == 'csv':
      csv_config = bq_pipeline_output_config.get('csv', {})
      # The inter-field delimiter for CVS (default ,) in the extract file
      extract_task_config['delimiter'] = csv_config.get('delimiter', ',')
      # Include a header (default True) in the extract file
      extract_task_config['header'] = csv_config.get('header', True)
      # Compress the extract file (default True)
      extract_task_config['compress'] = csv_config.get('compress', True)

    return extract_task_config
