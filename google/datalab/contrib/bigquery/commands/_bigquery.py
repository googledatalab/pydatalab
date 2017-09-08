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
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from builtins import str
from past.builtins import basestring

try:
  import IPython
  import IPython.core.display
  import IPython.core.magic
except ImportError:
  raise Exception('This module can only be loaded in ipython.')

import google.datalab.bigquery
import google.datalab.data
import google.datalab.utils
import google.datalab.utils.commands


def _create_pipeline_subparser(parser):
  pipeline_parser = parser.subcommand('pipeline', 'Creates a pipeline to execute a SQL query to '
                                                  'transform data using BigQuery.')

  # common arguments
  pipeline_parser.add_argument('-b', '--billing', type=int, help='BigQuery billing tier')
  pipeline_parser.add_argument('-n', '--name', type=str, help='BigQuery pipeline name')
  pipeline_parser.add_argument('-d', '--debug', action='store_true', default=False,
                               help='Print the airflow python spec.')

  return pipeline_parser


def _create_pipeline2_subparser(parser):
  pipeline_parser = parser.subcommand('pipeline2', 'Creates a pipeline to execute a SQL query to '
                                                   'transform data using BigQuery.')

  # common arguments
  pipeline_parser.add_argument('-b', '--billing', type=int, help='BigQuery billing tier')
  pipeline_parser.add_argument('-n', '--name', type=str, help='BigQuery pipeline name')
  pipeline_parser.add_argument('-d', '--debug', action='store_true', default=False,
                               help='Print the airflow python spec.')

  return pipeline_parser


def _construct_context_for_args(args):
  """Construct a new Context for the parsed arguments.

  Args:
    args: the dictionary of magic arguments.
  Returns:
    A new Context based on the current default context, but with any explicitly
      specified arguments overriding the default's config.
  """
  global_default_context = google.datalab.Context.default()
  config = {}
  for key in global_default_context.config:
    config[key] = global_default_context.config[key]

  billing_tier_arg = args.get('billing', None)
  if billing_tier_arg:
    config['bigquery_billing_tier'] = billing_tier_arg

  return google.datalab.Context(
    project_id=global_default_context.project_id,
    credentials=global_default_context.credentials,
    config=config)


def _get_query_argument(args, cell, env):
  """ Get a query argument to a cell magic.

  The query is specified with args['query']. We look that up and if it is a BQ query
  object, just return it. If it is a string, build a query object out of it and return
  that

  Args:
    args: the dictionary of magic arguments.
    cell: the cell contents which can be variable value overrides (if args has a 'query'
        value) or inline SQL otherwise.
    env: a dictionary that is used for looking up variable values.

  Returns:
    A Query object.
  """
  sql_arg = args.get('query', None)
  if sql_arg is None:
    # Assume we have inline SQL in the cell
    if not isinstance(cell, basestring):
      raise Exception('Expected a --query argument or inline SQL')
    return google.datalab.bigquery.Query(cell, env=env)

  item = google.datalab.utils.commands.get_notebook_item(sql_arg)
  if isinstance(item, google.datalab.bigquery.Query):
    return item
  else:
    raise Exception('Expected a query object, got %s.' % type(item))


def _get_query_parameters(args, cell_body):
  """Extract query parameters from cell body if provided
  Also validates the cell body schema using jsonschema to catch errors before sending the http
  request. This validation isn't complete, however; it does not validate recursive schemas,
  but it acts as a good filter against most simple schemas

  Args:
    args: arguments passed to the magic cell
    cell_body: body of the magic cell

  Returns:
    Validated object containing query parameters
  """

  env = google.datalab.utils.commands.notebook_environment()
  config = google.datalab.utils.commands.parse_config(cell_body, env=env, as_dict=False)
  sql = args['query']
  if sql is None:
    raise Exception('Cannot extract query parameters in non-query cell')

  # Validate query_params
  if config:
    jsonschema.validate(config, query_params_schema)

    # Parse query_params. We're exposing a simpler schema format than the one actually required
    # by BigQuery to make magics easier. We need to convert between the two formats
    parsed_params = []
    for param in config['parameters']:
      parsed_params.append({
        'name': param['name'],
        'parameterType': {
          'type': param['type']
        },
        'parameterValue': {
          'value': param['value']
        }
      })
    return parsed_params
  else:
    return {}


def _pipeline_cell(args, cell_body):
    """Implements the pipeline subcommand in the %%bq magic.

    The supported syntax is:

        %%bq pipeline <args>
        [<inline YAML>]

    Args:
      args: the arguments following '%%bq pipeline'.
      cell_body: the contents of the cell
    """
    name = args.get('name')
    if name is None:
        raise Exception("Pipeline name was not specified.")

    bq_pipeline_config = google.datalab.utils.commands.parse_config(
        cell_body, google.datalab.utils.commands.notebook_environment())

    load_task_config_name = 'bq_pipeline_load_task'
    load_task_config = {'type': 'pydatalab.bq.load'}
    add_load_parameters(load_task_config, bq_pipeline_config)

    execute_task_config_name = 'bq_pipeline_execute_task'
    execute_task_config = {'type': 'pydatalab.bq.execute', 'up_stream': [load_task_config_name]}
    add_execute_parameters(execute_task_config, bq_pipeline_config)

    extract_task_config_name = 'bq_pipeline_extract_task'
    extract_task_config = {'type': 'pydatalab.bq.extract', 'up_stream': [execute_task_config_name]}
    add_extract_parameters(extract_task_config, bq_pipeline_config)

    pipeline_spec = {
        'email': bq_pipeline_config['email'],
        'schedule': bq_pipeline_config['schedule'],
    }

    # These sections are only set when they aren't None
    pipeline_spec['tasks'] = {}
    if load_task_config:
        pipeline_spec['tasks'][load_task_config_name] = load_task_config
    if execute_task_config:
        pipeline_spec['tasks'][execute_task_config_name] = execute_task_config
    if extract_task_config:
        pipeline_spec['tasks'][extract_task_config_name] = extract_task_config

    if not load_task_config and not execute_task_config and not extract_task_config:
        raise Exception('Pipeline has no tasks to execute.')

    pipeline = google.datalab.contrib.pipeline._pipeline.Pipeline(name, pipeline_spec)
    google.datalab.utils.commands.notebook_environment()[name] = pipeline

    debug = args.get('debug')
    if debug is True:
        return pipeline.py


def _pipeline2_cell(args, cell_body):
    """Implements the pipeline subcommand in the %%bq magic.

    The supported syntax is:

        %%bq pipeline <args>
        [<inline YAML>]

    Args:
      args: the arguments following '%%bq pipeline'.
      cell_body: the contents of the cell
    """
    name = args.get('name')
    if name is None:
        raise Exception("Pipeline name was not specified.")

    bq_pipeline_config = google.datalab.utils.commands.parse_config(
        cell_body, google.datalab.utils.commands.notebook_environment())

    load_task_config_name = 'bq_pipeline_load_task'
    load_task_config = {'type': 'pydatalab.bq.load'}
    add_load_parameters2(load_task_config, bq_pipeline_config['input'])

    execute_task_config_name = 'bq_pipeline_execute_task'
    execute_task_config = {'type': 'pydatalab.bq.execute', 'up_stream': [load_task_config_name]}
    add_execute_parameters2(execute_task_config, bq_pipeline_config['input'])

    extract_task_config_name = 'bq_pipeline_extract_task'
    extract_task_config = {'type': 'pydatalab.bq.extract', 'up_stream': [execute_task_config_name]}
    add_extract_parameters2(extract_task_config, bq_pipeline_config['output'])

    pipeline_spec = {
        'email': bq_pipeline_config['email'],
        'schedule': bq_pipeline_config['schedule'],
    }

    # These sections are only set when they aren't None
    pipeline_spec['tasks'] = {}
    if load_task_config:
        pipeline_spec['tasks'][load_task_config_name] = load_task_config
    if execute_task_config:
        pipeline_spec['tasks'][execute_task_config_name] = execute_task_config
    if extract_task_config:
        pipeline_spec['tasks'][extract_task_config_name] = extract_task_config

    if not load_task_config and not execute_task_config and not extract_task_config:
        raise Exception('Pipeline has no tasks to execute.')

    pipeline = google.datalab.contrib.pipeline._pipeline.Pipeline(name, pipeline_spec)
    google.datalab.utils.commands.notebook_environment()[name] = pipeline

    debug = args.get('debug')
    if debug is True:
        return pipeline.py


def add_load_parameters(load_task_config, bq_pipeline_config):
    # One of 'csv' (default) or 'json' for the format of the load file
    load_task_config['format'] = bq_pipeline_config.get('load_format', 'csv')
    # The inter-field delimiter for CVS (default ,) in the load file
    load_task_config['delimiter'] = bq_pipeline_config.get('load_delimiter', ',')
    # One of 'create' (default), 'append' or 'overwrite' for loading data into BigQuery
    load_task_config['mode'] = bq_pipeline_config.get('load_mode', 'create')
    # The path URL of the GCS load file(s); if absent, we return None as there is
    # nothing to load
    if 'load_path' in bq_pipeline_config:
        load_task_config['path'] = bq_pipeline_config['load_path']
    else:
        return None
    # The quoted field delimiter for CVS (default ") in the load file
    load_task_config['quote'] = bq_pipeline_config.get('quote', '"')
    # The schema of the destination bigquery table
    load_task_config['schema'] = bq_pipeline_config['schema']
    # The number of head lines (default is 0) to skip during load; useful for CSV
    load_task_config['skip'] = bq_pipeline_config.get('skip', 0)
    # Reject bad values and jagged lines when loading (default True)
    load_task_config['strict'] = bq_pipeline_config.get('strict', True)
    # The destination bigquery table name for loading; if absent, we return None as there is
    # nothing to load
    if 'load_table' in bq_pipeline_config:
        load_task_config['table'] = bq_pipeline_config['load_table']
    else:
        return None
    # TODO(rajivpb): Consider raising an exception if 'path' is present and 'table' is not


def add_execute_parameters(execute_task_config, bq_pipeline_config):
    # Allow large results during execution; defaults to True because this is a common in pipelines
    execute_task_config['large'] = bq_pipeline_config.get('large', True)
    # One of 'create' (default), 'append' or 'overwrite' for the destination table in BigQuery
    execute_task_config['mode'] = bq_pipeline_config.get('execute_mode', 'create')
    # The name of query for execution; if absent, we return None as we assume that there is
    # no query to execute
    if 'query' in bq_pipeline_config:
        execute_task_config['query'] = bq_pipeline_config['query']
    else:
        return None
    # Destination table name for the execution results; defaults to None as this is
    # not required (the user may just want to execute a query)
    execute_task_config['table'] = bq_pipeline_config.get('execute_table', None)


def add_extract_parameters(extract_task_config, bq_pipeline_config):
    # TODO(rajivpb): The billing parameter should really be an arg and not in the yaml cell_body
    extract_task_config['billing'] = bq_pipeline_config['billing']
    # Compress the extract file (default True)
    extract_task_config['compress'] = bq_pipeline_config.get('compress', True)
    # The inter-field delimiter for CVS (default ,) in the extract file
    extract_task_config['delimiter'] = bq_pipeline_config.get('extract_delimiter', ',')
    # Include a header (default True) in the extract file
    extract_task_config['header'] = bq_pipeline_config.get('header', True)
    # The source table for the extract operation is the destination of the execute operation; if
    # absent we return None since we assume that there is no extract step.
    if 'execute_table' in bq_pipeline_config:
        extract_task_config['table'] = bq_pipeline_config['execute_table']
    else:
        return None
    # The destination GCS path for the extract file; if absent we return None since we assume that
    # there is no extract step.
    if 'extract_path' in bq_pipeline_config:
        extract_task_config['path'] = bq_pipeline_config['extract_path']
    else:
        return None
    # One of 'csv' (default) or 'json' for the format of the extract file
    extract_task_config['format'] = bq_pipeline_config.get('extract_format', 'csv')


def add_load_parameters2(load_task_config, bq_pipeline_config_input):
    # One of 'csv' (default) or 'json' for the format of the load file
    load_task_config['format'] = bq_pipeline_config.get('load_format', 'csv')
    # The inter-field delimiter for CVS (default ,) in the load file
    load_task_config['delimiter'] = bq_pipeline_config.get('load_delimiter', ',')
    # One of 'create' (default), 'append' or 'overwrite' for loading data into BigQuery
    load_task_config['mode'] = bq_pipeline_config.get('load_mode', 'create')
    # The path URL of the GCS load file(s); if absent, we return None as there is
    # nothing to load
    if 'load_path' in bq_pipeline_config:
        load_task_config['path'] = bq_pipeline_config['load_path']
    else:
        return None
    # The quoted field delimiter for CVS (default ") in the load file
    load_task_config['quote'] = bq_pipeline_config.get('quote', '"')
    # The schema of the destination bigquery table
    load_task_config['schema'] = bq_pipeline_config['schema']
    # The number of head lines (default is 0) to skip during load; useful for CSV
    load_task_config['skip'] = bq_pipeline_config.get('skip', 0)
    # Reject bad values and jagged lines when loading (default True)
    load_task_config['strict'] = bq_pipeline_config.get('strict', True)
    # The destination bigquery table name for loading; if absent, we return None as there is
    # nothing to load
    if 'load_table' in bq_pipeline_config:
        load_task_config['table'] = bq_pipeline_config['load_table']
    else:
        return None
    # TODO(rajivpb): Consider raising an exception if 'path' is present and 'table' is not



def add_execute_parameters2(execute_task_config, bq_pipeline_config_input):
    pass


def add_extract_parameters2(extract_task_config, bq_pipeline_config_input):
    pass