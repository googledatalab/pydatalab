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
  pipeline_parser.add_argument('-n', '--name', type=str, help='BigQuery pipeline name',
                               required=True)
  pipeline_parser.add_argument('-e', '--environment', type=str,
                               help='The name of the Composer or Airflow environment.')
  pipeline_parser.add_argument('-z', '--zone', type=str,
                               help='The name of the Composer or Airflow zone.')
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
        raise Exception('Pipeline name was not specified.')

    bq_pipeline_config = utils.commands.parse_config(
        cell_body, utils.commands.notebook_environment())
    pipeline_spec = _get_pipeline_spec_from_config(bq_pipeline_config)
    # TODO(rajivpb): This import is a stop-gap for
    # https://github.com/googledatalab/pydatalab/issues/593
    import google.datalab.contrib.pipeline._pipeline
    pipeline = google.datalab.contrib.pipeline._pipeline.Pipeline(name, pipeline_spec)
    utils.commands.notebook_environment()[name] = pipeline

    # If a composer environment and zone are specified, we deploy to composer
    if 'environment' in args and 'zone' in args:
        # TODO(rajivpb): This import is a stop-gap for
        # https://github.com/googledatalab/pydatalab/issues/593
        import google.datalab.contrib.pipeline.composer._composer
        composer = google.datalab.contrib.pipeline.composer._composer.Composer(
            args.get('zone'), args.get('environment'))
        composer.deploy(name, pipeline._get_airflow_spec())

    # TODO(rajivpb): See https://github.com/googledatalab/pydatalab/issues/501. Don't return python.
    return pipeline._get_airflow_spec()


def _get_pipeline_spec_from_config(bq_pipeline_config):
  pipeline_spec = {}

  schedule_config = bq_pipeline_config.get('schedule')
  if schedule_config:
    pipeline_spec['schedule'] = schedule_config

  email_config = bq_pipeline_config.get('emails')
  if email_config:
    pipeline_spec['emails'] = email_config

  input_config = bq_pipeline_config.get('input')
  transformation_config = bq_pipeline_config.get('transformation')
  output_config = bq_pipeline_config.get('output')
  parameters_config = bq_pipeline_config.get('parameters')

  pipeline_spec['tasks'] = {}

  load_task_id = None
  load_task_config = _get_load_parameters(input_config, transformation_config, output_config)
  if load_task_config:
    load_task_id = 'bq_pipeline_load_task'
    pipeline_spec['tasks'][load_task_id] = load_task_config

  execute_task_config = _get_execute_parameters(load_task_id, input_config, transformation_config,
                                                output_config, parameters_config)
  execute_task_id = None
  if execute_task_config:
    execute_task_id = 'bq_pipeline_execute_task'
    pipeline_spec['tasks'][execute_task_id] = execute_task_config

  extract_task_config = _get_extract_parameters(execute_task_id, input_config,
                                                transformation_config, output_config)
  if extract_task_config:
    pipeline_spec['tasks']['bq_pipeline_extract_task'] = extract_task_config

  if not load_task_config and not execute_task_config and not extract_task_config:
    raise Exception('Pipeline has no tasks to execute.')

  return pipeline_spec


def _get_load_parameters(bq_pipeline_input_config, bq_pipeline_transformation_config,
                         bq_pipeline_output_config):
    if bq_pipeline_input_config is None:
      return None

    load_task_config = {'type': 'pydatalab.bq.load'}

    # The path URL of the GCS load file(s).
    if 'path' not in bq_pipeline_input_config:
      return None

    # The path URL of the GCS load file(s), and associated parameters
    load_task_config['path'] = bq_pipeline_input_config.get('path')

    if 'format' in bq_pipeline_input_config:
      load_task_config['format'] = bq_pipeline_input_config['format']

    if 'csv' in bq_pipeline_input_config:
      load_task_config['csv_options'] = bq_pipeline_input_config['csv']

    # The destination bigquery table name for loading
    source_of_table = bq_pipeline_input_config
    if ('table' not in bq_pipeline_input_config and not bq_pipeline_transformation_config and
        bq_pipeline_output_config and 'table' in bq_pipeline_output_config and
            'path' not in bq_pipeline_output_config):
      # If we're here it means that there was no transformation config, but there was an output
      # config with only a table (and no path). We assume that the user was just trying to do a
      # gcs->table (or load) step, so we take that as the input table (and emit a load
      # operator).
      source_of_table = bq_pipeline_output_config

    # If a table or path are absent, there is no load to be done so we return None
    if 'table' not in source_of_table:
      return None

    load_task_config['table'] = source_of_table.get('table')

    if 'schema' in source_of_table:
      load_task_config['schema'] = source_of_table['schema']

    if 'mode' in source_of_table:
      load_task_config['mode'] = source_of_table['mode']

    return load_task_config


def _get_execute_parameters(load_task_id, bq_pipeline_input_config,
                            bq_pipeline_transformation_config, bq_pipeline_output_config,
                            bq_pipeline_parameters_config):
    if bq_pipeline_transformation_config is None:
      return None

    # The name of query for execution; if absent, we return None as we assume that there is
    # no query to execute
    if 'query' not in bq_pipeline_transformation_config:
      return None

    execute_task_config = {
      'type': 'pydatalab.bq.execute',
    }

    if load_task_id:
      execute_task_config['up_stream'] = [load_task_id]

    # Stuff from the input config
    if bq_pipeline_input_config and 'data_source' in bq_pipeline_input_config:
        execute_task_config['data_source'] = bq_pipeline_input_config['data_source']

        # All the below are applicable only if data_source is specified
        if 'path' in bq_pipeline_input_config:
            execute_task_config['path'] = bq_pipeline_input_config['path']

        if 'schema' in bq_pipeline_input_config:
            execute_task_config['schema'] = bq_pipeline_input_config['schema']

        if 'max_bad_records' in bq_pipeline_input_config:
            execute_task_config['max_bad_records'] = bq_pipeline_input_config['max_bad_records']

        if 'format' in bq_pipeline_input_config:
          execute_task_config['source_format'] = bq_pipeline_input_config.get('format')

        if 'csv' in bq_pipeline_input_config:
          execute_task_config['csv_options'] = bq_pipeline_input_config.get('csv')

    query = utils.commands.get_notebook_item(bq_pipeline_transformation_config['query'])
    execute_task_config['sql'] = query.sql

    # Stuff from the output config
    if bq_pipeline_output_config:
      if 'table' in bq_pipeline_output_config:
        execute_task_config['table'] = bq_pipeline_output_config['table']

      if 'mode' in bq_pipeline_output_config:
          execute_task_config['mode'] = bq_pipeline_output_config['mode']

    if bq_pipeline_parameters_config:
      execute_task_config['parameters'] = bq_pipeline_parameters_config

    return execute_task_config


def _get_extract_parameters(execute_task_id, bq_pipeline_input_config,
                            bq_pipeline_transformation_config, bq_pipeline_output_config):
    if bq_pipeline_output_config is None:
      return None

    extract_task_config = {
      'type': 'pydatalab.bq.extract',
    }

    if execute_task_id:
      extract_task_config['up_stream'] = [execute_task_id]

    # If a path is not specified, there is no extract to be done, so we return None
    if 'path' not in bq_pipeline_output_config:
      return None

    extract_task_config['path'] = bq_pipeline_output_config.get('path')

    if 'format' in bq_pipeline_output_config:
      extract_task_config['format'] = bq_pipeline_output_config.get('format')

    if 'csv' in bq_pipeline_output_config:
      extract_task_config['csv_options'] = bq_pipeline_output_config.get('csv')

    # If a temporary table from the bigquery results is being used, this will not be present in the
    # output section.
    if 'table' in bq_pipeline_output_config:
      extract_task_config['table'] = bq_pipeline_output_config['table']
    elif (bq_pipeline_input_config and not bq_pipeline_transformation_config and
          'table' in bq_pipeline_input_config and 'path' not in bq_pipeline_input_config):
      # If we're here it means that there was no transformation config, but there was an input
      # config with only a table (and no path). We assume that the user was just trying to do a
      # table->gcs (or extract) step, so we take that as the input table (and emit an extract
      # operator).
      extract_task_config['table'] = bq_pipeline_input_config['table']

    return extract_task_config
