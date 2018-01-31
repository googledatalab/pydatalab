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
import google
import google.datalab.utils as utils
import google.datalab.contrib.pipeline._pipeline

import jsonschema


def get_airflow_spec_from_config(name, bq_pipeline_config):
  pipeline_spec = google.datalab.contrib.bigquery.commands._bigquery._get_pipeline_spec_from_config(
    bq_pipeline_config)
  return google.datalab.contrib.pipeline._pipeline.PipelineGenerator.generate_airflow_spec(
    name, pipeline_spec)


def _get_pipeline_spec_from_config(bq_pipeline_config):
  pipeline_spec = {}

  schedule_config = bq_pipeline_config.get('schedule')
  if schedule_config:
    pipeline_spec['schedule'] = schedule_config

  email_config = bq_pipeline_config.get('emails')
  if email_config:
    pipeline_spec['emails'] = email_config

  input_config = bq_pipeline_config.get('input') or bq_pipeline_config.get('load')
  transformation_config = bq_pipeline_config.get('transformation')
  output_config = bq_pipeline_config.get('output') or bq_pipeline_config.get('extract')

  parameters_config = bq_pipeline_config.get('parameters')
  if parameters_config:
    jsonschema.validate(
      {'parameters': parameters_config},
      google.datalab.bigquery.commands._bigquery.BigQuerySchema.QUERY_PARAMS_SCHEMA)
  pipeline_spec['parameters'] = parameters_config

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

    # The destination BQ table name for loading
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

    # If the input config has a path but no table, we assume that the user has specified an
    # external data_source either explicitly (i.e. via specifying a "data_source" key in the input
    # config, or implicitly (i.e. by letting us assume that this is called "input")
    if (bq_pipeline_input_config and 'path' in bq_pipeline_input_config and
            'table' not in bq_pipeline_input_config):
        execute_task_config['data_source'] = bq_pipeline_input_config.get('data_source', 'input')

        if 'path' in bq_pipeline_input_config:
            # We format the path since this could contain format modifiers
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
    # If there is a table in the input config, we allow the user to reference table with the name
    # 'input' in their sql, i.e. via something like 'SELECT col1 FROM input WHERE ...'. To enable
    # this, we include the input table as a subquery with the query object. If the user's sql does
    # not reference an 'input' table, BigQuery will just ignore it. Things get interesting if the
    # user's sql specifies a subquery named 'input' - that should override the subquery that we use.
    # TODO(rajivpb): Verify this.
    if (bq_pipeline_input_config and 'table' in bq_pipeline_input_config):
      table_name = google.datalab.bigquery.Query.resolve_parameters(
          bq_pipeline_input_config.get('table'), bq_pipeline_parameters_config, macros=True)
      input_subquery_sql = 'SELECT * FROM `{0}`'.format(table_name)
      input_subquery = google.datalab.bigquery.Query(input_subquery_sql)
      # We artificially create an env with just the 'input' key, and the new input_query value to
      # fool the Query object into using the subquery correctly.
      query = google.datalab.bigquery.Query(query.sql, env={'input': input_subquery},
                                            subqueries=['input'])

    execute_task_config['sql'] = query.sql
    execute_task_config['parameters'] = bq_pipeline_parameters_config

    if bq_pipeline_output_config:
      if 'table' in bq_pipeline_output_config:
        execute_task_config['table'] = bq_pipeline_output_config['table']

      if 'mode' in bq_pipeline_output_config:
          execute_task_config['mode'] = bq_pipeline_output_config['mode']

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
      extract_task_config['table'] = """{{{{ ti.xcom_pull(task_ids='{0}_id').get('table') }}}}"""\
          .format(execute_task_id)

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
    source_of_table = None
    if 'table' in bq_pipeline_output_config:
      source_of_table = bq_pipeline_output_config
    elif (bq_pipeline_input_config and not bq_pipeline_transformation_config and
          'table' in bq_pipeline_input_config and 'path' not in bq_pipeline_input_config):
      # If we're here it means that there was no transformation config, but there was an input
      # config with only a table and no path. We assume that the user was just trying to do a
      # table->gcs (or extract) step, so we take that as the input table (and emit an extract
      # operator).
      source_of_table = bq_pipeline_input_config

    if source_of_table:
      extract_task_config['table'] = source_of_table['table']

    return extract_task_config
