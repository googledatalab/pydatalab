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
from builtins import zip
from builtins import str
from past.builtins import basestring

try:
  import IPython
  import IPython.core.display
  import IPython.core.magic
except ImportError:
  raise Exception('This module can only be loaded in ipython.')

import jsonschema
import fnmatch
import json
import re

import google.datalab.bigquery
from google.datalab.bigquery._query_output import QueryOutput
from google.datalab.bigquery._sampling import Sampling
import google.datalab.data
import google.datalab.utils
import google.datalab.utils.commands

BIGQUERY_DATATYPES = ['STRING', 'BYTES', 'INTEGER', 'INT64', 'FLOAT', 'FLOAT64', 'BOOLEAN',
                      'BOOL', 'TIMESTAMP', 'DATE', 'TIME', 'DATETIME', 'RECORD']
BIGQUERY_DATATYPES_LOWER = [t.lower() for t in BIGQUERY_DATATYPES]
BIGQUERY_MODES = ['NULLABLE', 'REQUIRED', 'REPEATED']
BIGQUERY_MODES_LOWER = [m.lower() for m in BIGQUERY_MODES]

table_schema_schema = {
  'definitions': {
    'field': {
      'title': 'field',
      'type': 'object',
      'properties': {
        'name': {'type': 'string'},
        'type': {'type': 'string', 'enum': BIGQUERY_DATATYPES + BIGQUERY_DATATYPES_LOWER},
        'mode': {'type': 'string', 'enum': BIGQUERY_MODES + BIGQUERY_MODES_LOWER},
        'description': {'type': 'string'},
        'fields': {'$ref': '#/definitions/field'}
      },
      'required': ['name', 'type'],
      'additionalProperties': False
    }
  },
  'type': 'object',
  'properties': {
    'schema': {
      'type': 'array',
      'items': {
        'allOf': [{'$ref': '#/definitions/field'}]
      }
    }
  },
  'required': ['schema'],
  'additionalProperties': False
}

query_params_schema = {
  'type': 'object',
  'properties': {
    'parameters': {
      'type': 'array',
      'items': [
        {
          'type': 'object',
          'properties': {
            'name': {'type': 'string'},
            'type': {'type': 'string', 'enum': ['STRING', 'INT64']},
            'value': {'type': ['string', 'integer']}
          },
          'required': ['name', 'type', 'value'],
          'additionalProperties': False
        }
      ]
    }
  },
  'required': ['parameters'],
  'additionalProperties': False
}


def _create_dataset_subparser(parser):
  dataset_parser = parser.subcommand('datasets', 'Operations on BigQuery datasets')
  sub_commands = dataset_parser.add_subparsers(dest='command')

  # %%bq datasets list
  list_parser = sub_commands.add_parser('list', help='List datasets')
  list_parser.add_argument('-p', '--project',
                           help='The project whose datasets should be listed')
  list_parser.add_argument('-f', '--filter',
                           help='Optional wildcard filter string used to limit the results')

  # %%bq datasets create
  create_parser = sub_commands.add_parser('create', help='Create a dataset.')
  create_parser.add_argument('-n', '--name', help='The name of the dataset to create.',
                                     required=True)
  create_parser.add_argument('-f', '--friendly', help='The friendly name of the dataset.')

  # %%bq datasets delete
  delete_dataset_parser = sub_commands.add_parser('delete', help='Delete a dataset.')
  delete_dataset_parser.add_argument('-n', '--name', help='The name of the dataset to delete.',
                                     required=True)

  return dataset_parser


def _create_table_subparser(parser):
  table_parser = parser.subcommand('tables', 'Operations on BigQuery tables')
  sub_commands = table_parser.add_subparsers(dest='command')

  # %%bq tables list
  list_parser = sub_commands.add_parser('list', help='List the tables in a BigQuery project or dataset.')
  list_parser.add_argument('-p', '--project',
                             help='The project whose tables should be listed')
  list_parser.add_argument('-d', '--dataset',
                             help='The dataset to restrict to')
  list_parser.add_argument('-f', '--filter',
                             help='Optional wildcard filter string used to limit the results')

  # %%bq tables create
  create_parser = sub_commands.add_parser('create', help='Create a table.')
  create_parser.add_argument('-n', '--name', help='The name of the table to create.',
                                   required=True)
  create_parser.add_argument('-o', '--overwrite', help='Overwrite table if it exists.',
                                   action='store_true')

  # %%bq tables describe
  describe_parser = sub_commands.add_parser('describe', help='View a table\'s schema')
  describe_parser.add_argument('-n', '--name', help='Name of table to show', required=True)

  # %%bq tables delete
  delete_parser = sub_commands.add_parser('delete', help='Delete a table.')
  delete_parser.add_argument('-n', '--name', help='The name of the table to delete.',
                                   required=True)

  # %%bq tables view
  delete_parser = sub_commands.add_parser('view', help='View a table.')
  delete_parser.add_argument('-n', '--name', help='The name of the table to view.',
                                   required=True)

  return table_parser


def _create_sample_subparser(parser):
  sample_parser = parser.subcommand('sample', help='Display a sample of the results of a ' +
      'BigQuery SQL query. The cell can optionally contain arguments for expanding variables in ' +
      'the query, if -q/--query was used, or it can contain SQL for a query.')
  group = sample_parser.add_mutually_exclusive_group()
  group.add_argument('-q', '--query', help='the name of the query object to sample')
  group.add_argument('-t', '--table', help='the name of the table object to sample')
  group.add_argument('-v', '--view', help='the name of the view object to sample')
  sample_parser.add_argument('-nc', '--nocache', help='Don\'t use previously cached results',
                              action='store_true')
  sample_parser.add_argument('-b', '--billing', type=int, help='BigQuery billing tier')
  sample_parser.add_argument('-m', '--method', help='The type of sampling to use',
                             choices=['limit', 'random', 'hashed', 'sorted'], default='limit')
  sample_parser.add_argument('--fields', help='Comma separated field names for projection')
  sample_parser.add_argument('-c', '--count', type=int, default=10,
                             help='The number of rows to limit to, if sampling')
  sample_parser.add_argument('-p', '--percent', type=int, default=1,
                             help='For random or hashed sampling, what percentage to sample from')
  sample_parser.add_argument('--key-field',
                             help='The field to use for sorted or hashed sampling')
  sample_parser.add_argument('-o', '--order', choices=['ascending', 'descending'],
                             default='ascending', help='The sort order to use for sorted sampling')
  sample_parser.add_argument('-P', '--profile', action='store_true',
                             default=False, help='Generate an interactive profile of the data')
  sample_parser.add_argument('--verbose',
                             help='Show the expanded SQL that is being executed',
                             action='store_true')
  return sample_parser


def _create_udf_subparser(parser):
  udf_parser = parser.subcommand('udf', 'Create a named Javascript BigQuery UDF')
  udf_parser.add_argument('-n', '--name', help='The name for this UDF', required=True)
  udf_parser.add_argument('-l', '--language', help='The language of the function', required=True,
                          choices=['sql', 'js'])
  return udf_parser


def _create_datasource_subparser(parser):
  datasource_parser = parser.subcommand('datasource', 'Create a named Javascript BigQuery external data source')
  datasource_parser.add_argument('-n', '--name', help='The name for this data source',
                                 required=True)
  datasource_parser.add_argument('-p', '--paths', help='URL(s) of the data objects, can include ' + \
                                 'a wildcard "*" at the end', required=True, nargs='+')
  datasource_parser.add_argument('-f', '--format', help='The format of the table\'s data. ' + \
                                 'CSV or JSON, default CSV', default='CSV')
  datasource_parser.add_argument('-c', '--compressed', help='Whether the data is compressed',
                                 action='store_true')
  return datasource_parser


def _create_dryrun_subparser(parser):
  dryrun_parser = parser.subcommand('dryrun',
      'Execute a dry run of a BigQuery query and display approximate usage statistics')
  dryrun_parser.add_argument('-q', '--query',
                              help='The name of the query to be dry run')
  dryrun_parser.add_argument('-b', '--billing', type=int, help='BigQuery billing tier')
  dryrun_parser.add_argument('-v', '--verbose',
                              help='Show the expanded SQL that is being executed',
                              action='store_true')
  return dryrun_parser


def _create_query_subparser(parser):
  query_parser = parser.subcommand('query',
      'Create or execute a BigQuery SQL query object, optionally using other SQL objects, UDFs, ' + \
              'or external datasources. If a query name is not specified, the query is executed.')
  query_parser.add_argument('-n', '--name', help='The name of this SQL query object')
  query_parser.add_argument('--udfs', help='List of UDFs to reference in the query body', nargs='+')
  query_parser.add_argument('--datasources', help='List of external datasources to reference in the query body',
                            nargs='+')
  query_parser.add_argument('--subqueries', help='List of subqueries to reference in the query body', nargs='+')
  query_parser.add_argument('-v', '--verbose', help='Show the expanded SQL that is being executed',
                            action='store_true')
  return query_parser


def _create_execute_subparser(parser):
  execute_parser = parser.subcommand('execute',
      'Execute a BigQuery SQL query and optionally send the results to a named table.\n' +
      'The cell can optionally contain arguments for expanding variables in the query.')
  execute_parser.add_argument('-nc', '--nocache', help='Don\'t use previously cached results',
                              action='store_true')
  execute_parser.add_argument('-b', '--billing', type=int, help='BigQuery billing tier')
  execute_parser.add_argument('-m', '--mode', help='The table creation mode', default='create',
                              choices=['create', 'append', 'overwrite'])
  execute_parser.add_argument('-l', '--large', help='Whether to allow large results',
                              action='store_true')
  execute_parser.add_argument('-q', '--query', help='The name of query to run', required=True)
  execute_parser.add_argument('-t', '--table', help='Target table name')
  execute_parser.add_argument('--to-dataframe', help='Convert the result into a dataframe',
                              action='store_true')
  execute_parser.add_argument('--dataframe-start-row', help='Row of the table to start the ' +
                              'dataframe export')
  execute_parser.add_argument('--dataframe-max-rows', help='Upper limit on number of rows ' +
                              'to export to the dataframe', default=None)
  execute_parser.add_argument('-v', '--verbose',
                              help='Show the expanded SQL that is being executed',
                              action='store_true')
  return execute_parser


def _create_extract_subparser(parser):
  extract_parser = parser.subcommand('extract', 'Extract a query or table into file (local or GCS)')
  extract_parser.add_argument('-nc', '--nocache', help='Don\'t use previously cached results',
                              action='store_true')
  extract_parser.add_argument('-f', '--format', choices=['csv', 'json'], default='csv',
                              help='The format to use for the export')
  extract_parser.add_argument('-b', '--billing', type=int, help='BigQuery billing tier')
  extract_parser.add_argument('-c', '--compress', action='store_true',
                              help='Whether to compress the data')
  extract_parser.add_argument('-H', '--header', action='store_true',
                              help='Whether to include a header line (CSV only)')
  extract_parser.add_argument('-D', '--delimiter', default=',',
                              help='The field delimiter to use (CSV only)')
  group = extract_parser.add_mutually_exclusive_group()
  group.add_argument('-q', '--query', help='The name of query to extract')
  group.add_argument('-t', '--table', help='The name of the table to extract')
  group.add_argument('-v', '--view', help='The name of the view to extract')
  extract_parser.add_argument('-p', '--path', help='The path of the destination')
  extract_parser.add_argument('--verbose',
                              help='Show the expanded SQL that is being executed',
                              action='store_true')
  return extract_parser


def _create_load_subparser(parser):
  load_parser = parser.subcommand('load', 'Load data from GCS into a BigQuery table. If creating a new ' +
                                          'table, a schema should be specified in YAML or JSON in the cell ' +
                                          'body, otherwise the schema is inferred from existing table.')
  load_parser.add_argument('-m', '--mode', help='One of create (default), append or overwrite',
                           choices=['create', 'append', 'overwrite'], default='create')
  load_parser.add_argument('-f', '--format', help='The source format', choices=['json', 'csv'],
                           default='csv')
  load_parser.add_argument('--skip', help='The number of initial lines to skip; useful for CSV headers',
                           type=int, default=0)
  load_parser.add_argument('-s', '--strict', help='Whether to reject bad values and jagged lines',
                           action='store_true')
  load_parser.add_argument('-d', '--delimiter', default=',',
                           help='The inter-field delimiter for CVS (default ,)')
  load_parser.add_argument('-q', '--quote', default='"',
                           help='The quoted field delimiter for CVS (default ")')
  load_parser.add_argument('-p', '--path', help='The path URL of the GCS source(s)')
  load_parser.add_argument('-t', '--table', help='The destination table name')
  return load_parser


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

def _sample_cell(args, cell_body):
  """Implements the BigQuery sample magic for sampling queries
  The supported sytanx is:
    %%bq sample <args>
     [<inline SQL>]
  Args:
    args: the optional arguments following '%%bq sample'.
    cell_body: optional contents of the cell
  Returns:
    The results of executing the sampling query, or a profile of the sample data.
  """

  env = google.datalab.utils.commands.notebook_environment()
  query = None
  table = None
  view = None
  query_params = None

  if args['query']:
    query = google.datalab.utils.commands.get_notebook_item(args['query'])
    if query is None:
      raise Exception('Cannot find query %s.' % args['query'])
    query_params = _get_query_parameters(args, cell_body)

  elif args['table']:
    table = _get_table(args['table'])
    if not table:
      raise Exception('Could not find table %s' % args['table'])
  elif args['view']:
    view = google.datalab.utils.commands.get_notebook_item(args['view'])
    if not isinstance(view, google.datalab.bigquery.View):
      raise Exception('Could not find view %s' % args['view'])
  else:
    raise Exception('A query, table, or view is neede to sample')

  # parse comma-separated list of fields
  fields = args['fields'].split(',') if args['fields'] else None
  count = int(args['count']) if args['count'] else None
  percent = int(args['percent']) if args['percent'] else None
  sampling=Sampling._auto(method=args['method'], fields=fields, count=count,
                          percent=percent, key_field=args['key_field'],
                          ascending=args['order']=='ascending')

  context = _construct_context_for_args(args)

  if view:
    query = google.datalab.bigquery.Query.from_view(view)
  elif table:
    query = google.datalab.bigquery.Query.from_table(table)

  if args['profile']:
    results = query.execute(QueryOutput.dataframe(), sampling=sampling,
                            context=context, query_params=query_params).result()
  else:
    results = query.execute(QueryOutput.table(), sampling=sampling, context=context,
                            query_params=query_params).result()

  if args['verbose']:
    print(results.sql)

  if args['profile']:
    return google.datalab.utils.commands.profile_df(results)
  else:
    return results


def _dryrun_cell(args, cell_body):
  """Implements the BigQuery cell magic used to dry run BQ queries.

   The supported syntax is:
   %%bq dryrun [-q|--sql <query identifier>]
   [<YAML or JSON cell_body or inline SQL>]

  Args:
    args: the argument following '%bq dryrun'.
    cell_body: optional contents of the cell interpreted as YAML or JSON.
  Returns:
    The response wrapped in a DryRunStats object
  """
  query = _get_query_argument(args, cell_body, google.datalab.utils.commands.notebook_environment())

  if args['verbose']:
    print(query.sql)

  context = _construct_context_for_args(args)
  result = query.dry_run(context=context)
  return google.datalab.bigquery._query_stats.QueryStats(total_bytes=result['totalBytesProcessed'],
                                                         is_cached=result['cacheHit'])


def _udf_cell(args, cell_body):
  """Implements the Bigquery udf cell magic for ipython notebooks.

  The supported syntax is:
  %%bq udf --name <var> --language <lang>
  // @param <name> <type>
  // @returns <type>
  // @import <gcs_path>
  <js function>

  Args:
    args: the optional arguments following '%%bq udf'.
    cell_body: the UDF declaration (inputs and outputs) and implementation in javascript.
  """
  udf_name = args['name']
  if not udf_name:
    raise Exception('Declaration must be of the form %%bq udf --name <variable name>')

  # Parse out parameters, return type, and imports
  param_pattern = r'^\s*\/\/\s*@param\s+([<>\w]+)\s+([<>\w]+)\s*$'
  returns_pattern = r'^\s*\/\/\s*@returns\s+([<>\w]+)\s*$'
  import_pattern = r'^\s*\/\/\s*@import\s+(\S+)\s*$'

  params = re.findall(param_pattern, cell_body, re.MULTILINE)
  return_type = re.findall(returns_pattern, cell_body, re.MULTILINE)
  imports = re.findall(import_pattern, cell_body, re.MULTILINE)

  if len(return_type) < 1:
    raise Exception('UDF return type must be defined using // @returns <type>')
  if len(return_type) > 1:
    raise Exception('Found more than one return type definition')

  return_type = return_type[0]

  # Finally build the UDF object
  udf = google.datalab.bigquery.UDF(udf_name, cell_body, return_type, params, args['language'], imports)
  google.datalab.utils.commands.notebook_environment()[udf_name] = udf


def _datasource_cell(args, cell_body):
  """Implements the BigQuery datasource cell magic for ipython notebooks.

  The supported syntax is
  %%bq datasource --name <var> --paths <url> [--format <CSV|JSON>]
  <schema>

  Args:
    args: the optional arguments following '%%bq datasource'
    cell_body: the datasource's schema in json/yaml
  """
  name = args['name']
  paths = args['paths']
  data_format = (args['format'] or 'CSV').lower()
  compressed = args['compressed'] or False

  # Get the source schema from the cell body
  record = google.datalab.utils.commands.parse_config(cell_body,
                                   google.datalab.utils.commands.notebook_environment(),
                                   as_dict=False)

  jsonschema.validate(record, table_schema_schema)
  schema = google.datalab.bigquery.Schema(record['schema'])

  # Finally build the datasource object
  datasource = google.datalab.bigquery.ExternalDataSource(source=paths, source_format=data_format,
                                                          compressed=compressed, schema=schema)
  google.datalab.utils.commands.notebook_environment()[name] = datasource


def _query_cell(args, cell_body):
  """Implements the BigQuery cell magic for used to build SQL objects.

  The supported syntax is:

      %%bq query <args>
      [<inline SQL>]

  Args:
    args: the optional arguments following '%%bql query'.
    cell_body: the contents of the cell
  """
  name = args['name']
  udfs = args['udfs']
  datasources = args['datasources']
  subqueries = args['subqueries']

  # Finally build the query object
  query = google.datalab.bigquery.Query(cell_body, env=IPython.get_ipython().user_ns,
                                        udfs=udfs, data_sources=datasources, subqueries=subqueries)

  # if no name is specified, execute this query instead of defining it
  if name is None:
    return query.execute().result()
  else:
    google.datalab.utils.commands.notebook_environment()[name] = query


def _execute_cell(args, cell_body):
  """Implements the BigQuery cell magic used to execute BQ queries.

   The supported syntax is:
     %%bq execute <args>
     [<inline SQL>]

  Args:
    args: the optional arguments following '%%bq execute'.
    cell_body: optional contents of the cell
  Returns:
    QueryResultsTable containing query result
  """
  query = google.datalab.utils.commands.get_notebook_item(args['query'])
  if args['verbose']:
    print(query.sql)

  query_params = _get_query_parameters(args, cell_body)

  if args['to_dataframe']:
    # re-parse the int arguments because they're passed as strings
    start_row = int(args['dataframe_start_row']) if args['dataframe_start_row'] else None
    max_rows = int(args['dataframe_max_rows']) if args['dataframe_max_rows'] else None
    output_options = QueryOutput.dataframe(start_row=start_row, max_rows=max_rows,
                                           use_cache=not args['nocache'])
  else:
    output_options = QueryOutput.table(name=args['table'], mode=args['mode'],
                                       use_cache=not args['nocache'],
                                       allow_large_results=args['large'])
  context = _construct_context_for_args(args)
  r = query.execute(output_options, context=context, query_params=query_params)
  return r.result()


def _get_schema(name):
  """ Given a variable or table name, get the Schema if it exists. """
  item = google.datalab.utils.commands.get_notebook_item(name)
  if not item:
    item = _get_table(name)

  if isinstance(item, google.datalab.bigquery.Schema):
    return item
  if hasattr(item, 'schema') and isinstance(item.schema, google.datalab.bigquery._schema.Schema):
    return item.schema
  return None


# An LRU cache for Tables. This is mostly useful so that when we cross page boundaries
# when paging through a table we don't have to re-fetch the schema.
_table_cache = google.datalab.utils.LRUCache(10)


def _get_table(name):
  """ Given a variable or table name, get a Table if it exists.

  Args:
    name: the name of the Table or a variable referencing the Table.
  Returns:
    The Table, if found.
  """
  # If name is a variable referencing a table, use that.
  item = google.datalab.utils.commands.get_notebook_item(name)
  if isinstance(item, google.datalab.bigquery.Table):
    return item
  # Else treat this as a BQ table name and return the (cached) table if it exists.
  try:
    return _table_cache[name]
  except KeyError:
    table = google.datalab.bigquery.Table(name)
    if table.exists():
      _table_cache[name] = table
      return table
  return None


def _render_table(data, fields=None):
  """ Helper to render a list of dictionaries as an HTML display object. """
  return IPython.core.display.HTML(google.datalab.utils.commands.HtmlBuilder.render_table(data, fields))


def _render_list(data):
  """ Helper to render a list of objects as an HTML list object. """
  return IPython.core.display.HTML(google.datalab.utils.commands.HtmlBuilder.render_list(data))


def _dataset_line(args):
  """Implements the BigQuery dataset magic subcommand used to operate on datasets

   The supported syntax is:
   %bq datasets <command> <args>

  Commands:
    {list, create, delete}

  Args:
    args: the optional arguments following '%bq datasets command'.
  """
  if args['command'] == 'list':
    filter_ = args['filter'] if args['filter'] else '*'
    context = google.datalab.Context.default()
    if args['project']:
      context = google.datalab.Context(args['project'], context.credentials)
    return _render_list([str(dataset) for dataset in google.datalab.bigquery.Datasets(context)
                         if fnmatch.fnmatch(str(dataset), filter_)])

  elif args['command'] == 'create':
    try:
      google.datalab.bigquery.Dataset(args['name']).create(friendly_name=args['friendly'],
                                                    description=cell_body)
    except Exception as e:
      print('Failed to create dataset %s: %s' % (args['name'], e))

  elif args['command'] == 'delete':
    try:
      google.datalab.bigquery.Dataset(args['name']).delete()
    except Exception as e:
      print('Failed to delete dataset %s: %s' % (args['name'], e))


def _table_cell(args, cell_body):
  """Implements the BigQuery table magic subcommand used to operate on tables

   The supported syntax is:
   %%bq tables <command> <args>

  Commands:
    {list, create, delete, describe, view}

  Args:
    args: the optional arguments following '%%bq tables command'.
    cell_body: optional contents of the cell interpreted as SQL, YAML or JSON.
  Returns:
    The HTML rendering for the table of datasets.
  """
  if args['command'] == 'list':
    filter_ = args['filter'] if args['filter'] else '*'
    if args['dataset']:
      if args['project'] is None:
        datasets = [google.datalab.bigquery.Dataset(args['dataset'])]
      else:
        datasets = [google.datalab.bigquery.Dataset((args['project'], args['dataset']))]
    else:
      default_context = google.datalab.Context.default()
      context = google.datalab.Context(default_context.project_id, default_context.credentials)
      if args['project']:
        context.project_id = args['project']
      datasets = google.datalab.bigquery.Datasets(context)

    tables = []
    for dataset in datasets:
      tables.extend([str(table) for table in dataset if fnmatch.fnmatch(str(table), filter_)])

    return _render_list(tables)

  elif args['command'] == 'create':
    if cell_body is None:
      print('Failed to create %s: no schema specified' % args['name'])
    else:
      try:
        record = google.datalab.utils.commands.parse_config(cell_body,
                                         google.datalab.utils.commands.notebook_environment(),
                                         as_dict=False)
        jsonschema.validate(record, table_schema_schema)
        schema = google.datalab.bigquery.Schema(record['schema'])
        google.datalab.bigquery.Table(args['name']).create(schema=schema,
                                                    overwrite=args['overwrite'])
      except Exception as e:
        print('Failed to create table %s: %s' % (args['name'], e))

  elif args['command'] == 'describe':
    name = args['name']
    table = _get_table(name)
    if not table:
      raise Exception('Could not find table %s' % name)

    html = _repr_html_table_schema(table.schema)
    return IPython.core.display.HTML(html)

  elif args['command'] == 'delete':
    try:
      google.datalab.bigquery.Table(args['name']).delete()
    except Exception as e:
      print('Failed to delete table %s: %s' % (args['name'], e))

  elif args['command'] == 'view':
    name = args['name']
    table = _get_table(name)
    if not table:
      raise Exception('Could not find table %s' % name)
    return table

def _extract_cell(args, cell_body):
  """Implements the BigQuery extract magic used to extract query or table data to GCS.

   The supported syntax is:
     %bq extract <args>

  Args:
    args: the arguments following '%bigquery extract'.
  """
  env = google.datalab.utils.commands.notebook_environment()
  query = google.datalab.utils.commands.get_notebook_item(args['query'])
  query_params = _get_query_parameters(args, cell_body)

  if args['table'] or args['view']:
    if args['table']:
      source = _get_table(args['table'])
      if not source:
        raise Exception('Could not find table %s' % args['table'])
    elif args['view']:
      source = datalab.utils.commands.get_notebook_item(args['view'])
      if not source:
        raise Exception('Could not find view %' % args['view'])

    job = source.extract(args['path'],
                      format='CSV' if args['format'] == 'csv' else 'NEWLINE_DELIMITED_JSON',
                      csv_delimiter=args['delimiter'], csv_header=args['header'],
                      compress=args['compress'], use_cache=not args['nocache'])
  elif query:
    output_options = QueryOutput.file(path=args['path'], format=args['format'],
                                      csv_delimiter=args['delimiter'],
                                      csv_header=args['header'], compress=args['compress'],
                                      use_cache=not args['nocache'])
    context = _construct_context_for_args(args)
    job = query.execute(output_options, context=context, query_params=query_params)
  else:
    raise Exception('A query, table, or view is needed to extract')

  if job.failed:
    raise Exception('Extract failed: %s' % str(job.fatal_error))
  elif job.errors:
    raise Exception('Extract completed with errors: %s' % str(job.errors))
  return job.result()


def _load_cell(args, cell_body):
  """Implements the BigQuery load magic used to load data from GCS to a table.

   The supported syntax is:

       %bq load <optional args>

  Args:
    args: the arguments following '%bq load'.
    cell_body: optional contents of the cell interpreted as YAML or JSON.
  Returns:
    A message about whether the load succeeded or failed.
  """
  name = args['table']
  table = _get_table(name)
  if not table:
    table = google.datalab.bigquery.Table(name)

  if cell_body:
    schema = google.datalab.utils.commands.parse_config(cell_body, None, False)
    jsonschema.validate(schema, table_schema_schema)
    schema = google.datalab.bigquery.Schema(schema['schema'])
    table.create(schema=schema)
  elif table.exists():
    if args['mode'] == 'create':
      raise Exception('%s already exists; use --append or --overwrite' % name)
  else:
    raise Exception('Table does not exist, and no schema specified in cell; cannot load')

  csv_options = google.datalab.bigquery.CSVOptions(delimiter=args['delimiter'],
                                        skip_leading_rows=args['skip'],
                                        allow_jagged_rows=not args['strict'],
                                        quote=args['quote'])
  job = table.load(args['path'],
                   mode=args['mode'],
                   source_format=('csv' if args['format'] == 'csv' else 'NEWLINE_DELIMITED_JSON'),
                   csv_options=csv_options,
                   ignore_unknown_values=not args['strict'])
  if job.failed:
    raise Exception('Load failed: %s' % str(job.fatal_error))
  elif job.errors:
    raise Exception('Load completed with errors: %s' % str(job.errors))


def _add_command(parser, subparser_fn, handler, cell_required=False, cell_prohibited=False):
  """ Create and initialize a bigquery subcommand handler. """
  sub_parser = subparser_fn(parser)
  sub_parser.set_defaults(func=lambda args, cell: _dispatch_handler(args, cell, sub_parser, handler,
                          cell_required=cell_required, cell_prohibited=cell_prohibited))


def _create_bigquery_parser():
  """ Create the parser for the %bq magics.

  Note that because we use the func default handler dispatch mechanism of argparse,
  our handlers can take only one argument which is the parsed args. So we must create closures
  for the handlers that bind the cell contents and thus must recreate this parser for each
  cell upon execution.
  """
  parser = google.datalab.utils.commands.CommandParser(prog='bq', description="""
Execute various BigQuery-related operations. Use "%bq <command> -h"
for help on a specific command.
  """)

  # This is a bit kludgy because we want to handle some line magics and some cell magics
  # with the bq command.

  # %bq datasets
  _add_command(parser, _create_dataset_subparser, _dataset_line, cell_prohibited=True)

  # %bq tables
  _add_command(parser, _create_table_subparser, _table_cell)

  # %%bq query
  _add_command(parser, _create_query_subparser, _query_cell)

  # %%bq execute
  _add_command(parser, _create_execute_subparser, _execute_cell)

  # %bq extract
  _add_command(parser, _create_extract_subparser, _extract_cell)

  # %%bq sample
  _add_command(parser, _create_sample_subparser, _sample_cell)

  # %%bq dryrun
  _add_command(parser, _create_dryrun_subparser, _dryrun_cell)

  # %%bq udf
  _add_command(parser, _create_udf_subparser, _udf_cell, cell_required=True)

  # %%bq datasource
  _add_command(parser, _create_datasource_subparser, _datasource_cell, cell_required=True)

  # %bq load
  _add_command(parser, _create_load_subparser, _load_cell)
  return parser


_bigquery_parser = _create_bigquery_parser()


@IPython.core.magic.register_line_cell_magic
def bq(line, cell=None):
  """Implements the bq cell magic for ipython notebooks.

  The supported syntax is:

    %%bq <command> [<args>]
    <cell>

  or:

    %bq <command> [<args>]

  Use %bq --help for a list of commands, or %bq <command> --help for help
  on a specific command.
  """
  namespace = {}
  if line.find('$') >= 0:
    # We likely have variables to expand; get the appropriate context.
    namespace = google.datalab.utils.commands.notebook_environment()

  return google.datalab.utils.commands.handle_magic_line(line, cell, _bigquery_parser, namespace=namespace)


def _dispatch_handler(args, cell, parser, handler, cell_required=False, cell_prohibited=False):
  """ Makes sure cell magics include cell and line magics don't, before dispatching to handler.

  Args:
    args: the parsed arguments from the magic line.
    cell: the contents of the cell, if any.
    parser: the argument parser for <cmd>; used for error message.
    handler: the handler to call if the cell present/absent check passes.
    cell_required: True for cell magics, False for line magics that can't be cell magics.
    cell_prohibited: True for line magics, False for cell magics that can't be line magics.
  Returns:
    The result of calling the handler.
  Raises:
    Exception if the invocation is not valid.
  """
  if cell_prohibited:
    if cell and len(cell.strip()):
      parser.print_help()
      raise Exception('Additional data is not supported with the %s command.' % parser.prog)
    return handler(args)

  if cell_required and not cell:
    parser.print_help()
    raise Exception('The %s command requires additional data' % parser.prog)

  return handler(args, cell)


def _table_viewer(table, rows_per_page=25, fields=None):
  """  Return a table viewer.

    This includes a static rendering of the first page of the table, that gets replaced
    by the charting code in environments where Javascript is executable and BQ is available.

  Args:
    table: the table to view.
    rows_per_page: how many rows to display at one time.
    fields: an array of field names to display; default is None which uses the full schema.
  Returns:
    A string containing the HTML for the table viewer.
  """

  # TODO(gram): rework this to use google.datalab.utils.commands.chart_html

  if not table.exists():
    raise Exception('Table %s does not exist' % str(table))

  _HTML_TEMPLATE = u"""
    <div class="bqtv" id="{div_id}">{static_table}</div>
    <br />{meta_data}<br />
    <script>

      require.config({{
        paths: {{
          d3: '//cdnjs.cloudflare.com/ajax/libs/d3/3.4.13/d3',
          plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
          jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min'
        }},
        map: {{
          '*': {{
            datalab: 'nbextensions/gcpdatalab'
          }}
        }},
        shim: {{
          plotly: {{
            deps: ['d3', 'jquery'],
            exports: 'plotly'
          }}
        }}
      }});

      require(['datalab/charting', 'datalab/element!{div_id}', 'base/js/events',
          'datalab/style!/nbextensions/gcpdatalab/charting.css'],
        function(charts, dom, events) {{
          charts.render('gcharts', dom, events, '{chart_style}', [], {data},
            {{
              pageSize: {rows_per_page},
              cssClassNames:  {{
                tableRow: 'gchart-table-row',
                headerRow: 'gchart-table-headerrow',
                oddTableRow: 'gchart-table-oddrow',
                selectedTableRow: 'gchart-table-selectedrow',
                hoverTableRow: 'gchart-table-hoverrow',
                tableCell: 'gchart-table-cell',
                headerCell: 'gchart-table-headercell',
                rowNumberCell: 'gchart-table-rownumcell'
              }}
            }},
            {{source_index: {source_index}, fields: '{fields}'}},
            0,
            {total_rows});
        }}
      );
    </script>
  """

  if fields is None:
    fields = google.datalab.utils.commands.get_field_list(fields, table.schema)
  div_id = google.datalab.utils.commands.Html.next_id()
  meta_count = ('rows: %d' % table.length) if table.length >= 0 else ''
  meta_name = str(table) if table.job is None else ('job: %s' % table.job.id)
  if table.job:
    if table.job.cache_hit:
      meta_cost = 'cached'
    else:
      bytes = google.datalab.bigquery._query_stats.QueryStats._size_formatter(table.job.bytes_processed)
      meta_cost = '%s processed' % bytes
    meta_time = 'time: %.1fs' % table.job.total_time
  else:
    meta_cost = ''
    meta_time = ''

  data, total_count = google.datalab.utils.commands.get_data(table, fields, first_row=0, count=rows_per_page)

  if total_count < 0:
    # The table doesn't have a length metadata property but may still be small if we fetched less
    # rows than we asked for.
    fetched_count = len(data['rows'])
    if fetched_count < rows_per_page:
      total_count = fetched_count

  chart = 'table' if 0 <= total_count <= rows_per_page else 'paged_table'
  meta_entries = [meta_count, meta_time, meta_cost, meta_name]
  meta_data = '(%s)' % (', '.join([entry for entry in meta_entries if len(entry)]))

  return _HTML_TEMPLATE.format(div_id=div_id,
                               static_table=google.datalab.utils.commands.HtmlBuilder.render_chart_data(data),
                               meta_data=meta_data,
                               chart_style=chart,
                               source_index=google.datalab.utils.commands.get_data_source_index(str(table)),
                               fields=','.join(fields),
                               total_rows=total_count,
                               rows_per_page=rows_per_page,
                               data=json.dumps(data, cls=google.datalab.utils.JSONEncoder))


def _repr_html_query(query):
  # TODO(nikhilko): Pretty print the SQL
  return google.datalab.utils.commands.HtmlBuilder.render_text(query.sql, preformatted=True)


def _repr_html_query_results_table(results):
  return _table_viewer(results)


def _repr_html_table(results):
  return _table_viewer(results)


def _repr_html_table_schema(schema):
  _HTML_TEMPLATE = """
    <div class="bqsv" id="%s"></div>
    <script>
      require.config({
        map: {
          '*': {
            datalab: 'nbextensions/gcpdatalab'
          }
        },
      });

      require(['datalab/bigquery', 'datalab/element!%s',
          'datalab/style!/nbextensions/gcpdatalab/bigquery.css'],
        function(bq, dom) {
          bq.renderSchema(dom, %s);
        }
      );
    </script>
    """
  id = google.datalab.utils.commands.Html.next_id()
  return _HTML_TEMPLATE % (id, id, json.dumps(schema._bq_schema))


def _register_html_formatters():
  try:
    ipy = IPython.get_ipython()
    html_formatter = ipy.display_formatter.formatters['text/html']

    html_formatter.for_type_by_name('google.datalab.bigquery._query', 'Query', _repr_html_query)
    html_formatter.for_type_by_name('google.datalab.bigquery._query_results_table', 'QueryResultsTable',
                                    _repr_html_query_results_table)
    html_formatter.for_type_by_name('google.datalab.bigquery._table', 'Table', _repr_html_table)
    html_formatter.for_type_by_name('google.datalab.bigquery._schema', 'Schema', _repr_html_table_schema)
  except TypeError:
    # For when running unit tests
    pass

_register_html_formatters()
