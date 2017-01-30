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

import fnmatch
import json
import re

import google.datalab.bigquery
import google.datalab.data
import google.datalab.utils
import google.datalab.utils.commands


def _create_create_subparser(parser):
  create_parser = parser.subcommand('create', 'Create a dataset or table.')
  sub_commands = create_parser.add_subparsers(dest='command')

  create_dataset_parser = sub_commands.add_parser('dataset', help='Create a dataset.')
  create_dataset_parser.add_argument('-n', '--name', help='The name of the dataset to create.',
                                     required=True)
  create_dataset_parser.add_argument('-f', '--friendly', help='The friendly name of the dataset.')

  create_table_parser = sub_commands.add_parser('table', help='Create a table.')
  create_table_parser.add_argument('-n', '--name', help='The name of the table to create.',
                                   required=True)
  create_table_parser.add_argument('-o', '--overwrite', help='Overwrite table if it exists.',
                                   action='store_true')
  return create_parser


def _create_delete_subparser(parser):
  delete_parser = parser.subcommand('delete', 'Delete a dataset or table.')
  sub_commands = delete_parser.add_subparsers(dest='command')

  delete_dataset_parser = sub_commands.add_parser('dataset', help='Delete a dataset.')
  delete_dataset_parser.add_argument('-n', '--name', help='The name of the dataset to delete.',
                                     required=True)

  delete_table_parser = sub_commands.add_parser('table', help='Delete a table.')
  delete_table_parser.add_argument('-n', '--name', help='The name of the table to delete.',
                                   required=True)
  return delete_parser


def _create_sample_subparser(parser):
  sample_parser = parser.subcommand('sample',
      'Display a sample of the results of a BigQuery SQL query.\n' +
      'The cell can optionally contain arguments for expanding variables in the query,\n' +
      'if -q/--query was used, or it can contain SQL for a query.')
  group = sample_parser.add_mutually_exclusive_group()
  group.add_argument('-q', '--query', help='the name of the query to sample')
  group.add_argument('-t', '--table', help='the name of the table to sample')
  group.add_argument('-v', '--view', help='the name of the view to sample')
  sample_parser.add_argument('-d', '--dialect', help='BigQuery SQL dialect',
                             choices=['legacy', 'standard'])
  sample_parser.add_argument('-b', '--billing', type=int, help='BigQuery billing tier')
  sample_parser.add_argument('-c', '--count', type=int, default=10,
                             help='The number of rows to limit to, if sampling')
  sample_parser.add_argument('-m', '--method', help='The type of sampling to use',
                             choices=['limit', 'random', 'hashed', 'sorted'], default='limit')
  sample_parser.add_argument('-p', '--percent', type=int, default=1,
                             help='For random or hashed sampling, what percentage to sample from')
  sample_parser.add_argument('-f', '--field',
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
  udf_parser.add_argument('-m', '--module', help='The name for this UDF')
  return udf_parser


def _create_dry_run_subparser(parser):
  dry_run_parser = parser.subcommand('dryrun',
      'Execute a dry run of a BigQuery query and display approximate usage statistics')
  dry_run_parser.add_argument('-q', '--query',
                              help='The name of the query to be dry run')
  dry_run_parser.add_argument('-d', '--dialect', help='BigQuery SQL dialect',
                             choices=['legacy', 'standard'])
  dry_run_parser.add_argument('-b', '--billing', type=int, help='BigQuery billing tier')
  dry_run_parser.add_argument('-v', '--verbose',
                              help='Show the expanded SQL that is being executed',
                              action='store_true')
  return dry_run_parser


def _create_execute_subparser(parser):
  execute_parser = parser.subcommand('execute',
      'Execute a BigQuery SQL query and optionally send the results to a named table.\n' +
      'The cell can optionally contain arguments for expanding variables in the query.')
  execute_parser.add_argument('-nc', '--nocache', help='Don\'t use previously cached results',
                              action='store_true')
  execute_parser.add_argument('-d', '--dialect', help='BigQuery SQL dialect',
                             choices=['legacy', 'standard'])
  execute_parser.add_argument('-b', '--billing', type=int, help='BigQuery billing tier')
  execute_parser.add_argument('-m', '--mode', help='The table creation mode', default='create',
                              choices=['create', 'append', 'overwrite'])
  execute_parser.add_argument('-l', '--large', help='Whether to allow large results',
                              action='store_true')
  execute_parser.add_argument('-q', '--query', help='The name of query to run')
  execute_parser.add_argument('-t', '--target', help='target table name')
  execute_parser.add_argument('-v', '--verbose',
                              help='Show the expanded SQL that is being executed',
                              action='store_true')
  return execute_parser


def _create_pipeline_subparser(parser):
  pipeline_parser = parser.subcommand('pipeline',
      'Define a deployable pipeline based on a BigQuery query.\n' +
      'The cell can optionally contain arguments for expanding variables in the query.')
  pipeline_parser.add_argument('-n', '--name', help='The pipeline name')
  pipeline_parser.add_argument('-nc', '--nocache', help='Don\'t use previously cached results',
                               action='store_true')
  pipeline_parser.add_argument('-d', '--dialect', help='BigQuery SQL dialect',
                             choices=['legacy', 'standard'])
  pipeline_parser.add_argument('-b', '--billing', type=int, help='BigQuery billing tier')
  pipeline_parser.add_argument('-m', '--mode', help='The table creation mode', default='create',
                               choices=['create', 'append', 'overwrite'])
  pipeline_parser.add_argument('-l', '--large', help='Allow large results', action='store_true')
  pipeline_parser.add_argument('-q', '--query', help='The name of query to run', required=True)
  pipeline_parser.add_argument('-t', '--target', help='The target table name', nargs='?')
  pipeline_parser.add_argument('-v', '--verbose',
                               help='Show the expanded SQL that is being executed',
                               action='store_true')
  pipeline_parser.add_argument('action', nargs='?', choices=('deploy', 'run', 'dryrun'),
                               default='dryrun',
                               help='Whether to deploy the pipeline, execute it immediately in ' +
                                    'the notebook, or validate it with a dry run')
  # TODO(gram): we may want to move some command line arguments to the cell body config spec
  # eventually.
  return pipeline_parser


def _create_table_subparser(parser):
  table_parser = parser.subcommand('table', 'View a BigQuery table.')
  table_parser.add_argument('-r', '--rows', type=int, default=25,
                            help='Rows to display per page')
  table_parser.add_argument('-c', '--cols',
                            help='Comma-separated list of column names to restrict to')
  table_parser.add_argument('table', help='The name of, or a reference to, the table or view')
  return table_parser


def _create_schema_subparser(parser):
  schema_parser = parser.subcommand('schema', 'View a BigQuery table or view schema.')
  group = schema_parser.add_mutually_exclusive_group()
  group.add_argument('-v', '--view', help='the name of the view whose schema should be displayed')
  group.add_argument('-t', '--table', help='the name of the table whose schema should be displayed')
  return schema_parser


def _create_datasets_subparser(parser):
  datasets_parser = parser.subcommand('datasets', 'List the datasets in a BigQuery project.')
  datasets_parser.add_argument('-p', '--project',
                               help='The project whose datasets should be listed')
  datasets_parser.add_argument('-f', '--filter',
                               help='Optional wildcard filter string used to limit the results')
  return datasets_parser


def _create_tables_subparser(parser):
  tables_parser = parser.subcommand('tables', 'List the tables in a BigQuery project or dataset.')
  tables_parser.add_argument('-p', '--project',
                             help='The project whose tables should be listed')
  tables_parser.add_argument('-d', '--dataset',
                             help='The dataset to restrict to')
  tables_parser.add_argument('-f', '--filter',
                             help='Optional wildcard filter string used to limit the results')
  return tables_parser


def _create_extract_subparser(parser):
  extract_parser = parser.subcommand('extract', 'Extract BigQuery query results or table to GCS.')
  extract_parser.add_argument('-f', '--format', choices=['csv', 'json'], default='csv',
                              help='The format to use for the export')
  extract_parser.add_argument('-c', '--compress', action='store_true',
                              help='Whether to compress the data')
  extract_parser.add_argument('-H', '--header', action='store_true',
                              help='Whether to include a header line (CSV only)')
  extract_parser.add_argument('-d', '--delimiter', default=',',
                              help='The field delimiter to use (CSV only)')
  extract_parser.add_argument('-S', '--source', help='The name of the query or table to extract')
  extract_parser.add_argument('-D', '--destination', help='The URL of the destination')
  return extract_parser


def _create_load_subparser(parser):
  load_parser = parser.subcommand('load', 'Load data from GCS into a BigQuery table.')
  load_parser.add_argument('-m', '--mode', help='One of create (default), append or overwrite',
                           choices=['create', 'append', 'overwrite'], default='create')
  load_parser.add_argument('-f', '--format', help='The source format', choices=['json', 'csv'],
                           default='csv')
  load_parser.add_argument('-n', '--skip',
                           help='The number of initial lines to skip; useful for CSV headers',
                           type=int, default=0)
  load_parser.add_argument('-s', '--strict', help='Whether to reject bad values and jagged lines',
                           action='store_true')
  load_parser.add_argument('-d', '--delimiter', default=',',
                           help='The inter-field delimiter for CVS (default ,)')
  load_parser.add_argument('-q', '--quote', default='"',
                           help='The quoted field delimiter for CVS (default ")')
  load_parser.add_argument('-i', '--infer',
                           help='Whether to attempt to infer the schema from source; ' +
                               'if false the table must already exist',
                           action='store_true')
  load_parser.add_argument('-S', '--source', help='The URL of the GCS source(s)')
  load_parser.add_argument('-D', '--destination', help='The destination table name')
  return load_parser


def _get_query_argument(args, cell, env):
  """ Get a query argument to a cell magic.

  The query is specified with args['query']. We look that up and if it is a BQ query
  just return it. If it is instead a SqlModule or SqlStatement it may have variable
  references. We resolve those using the arg parser for the SqlModule, then override
  the resulting defaults with either the Python code in cell, or the dictionary in
  overrides. The latter is for if the overrides are specified with YAML or JSON and
  eventually we should eliminate code in favor of this.

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
    return google.datalab.bigquery.Query(cell, values=env)

  item = google.datalab.utils.commands.get_notebook_item(sql_arg)
  if isinstance(item, google.datalab.bigquery.Query):  # Queries are already expanded.
    return item

  # Create an expanded BQ Query.
  config = google.datalab.utils.commands.parse_config(cell, env)
  item, env = google.datalab.data.SqlModule.get_sql_statement_with_environment(item, config)
  if cell:
    env.update(config)  # config is both a fallback and an override.
  return google.datalab.bigquery.Query(item, values=env)


def _sample_cell(args, cell_body):
  """Implements the BigQuery sample cell magic for ipython notebooks.

  Args:
    args: the optional arguments following '%%bq sample'.
    cell_body: optional contents of the cell interpreted as SQL, YAML or JSON.
  Returns:
    The results of executing the sampling query, or a profile of the sample data.
  """

  env = google.datalab.utils.commands.notebook_environment()
  query = None
  table = None
  view = None

  if args['query']:
    query = _get_query_argument(args, cell_body, env)
  elif args['table']:
    table = _get_table(args['table'])
  elif args['view']:
    view = google.datalab.utils.commands.get_notebook_item(args['view'])
    if not isinstance(view, google.datalab.bigquery.View):
      raise Exception('%s is not a view' % args['view'])
  else:
    query = google.datalab.bigquery.Query(cell_body, values=env)

  count = args['count']
  method = args['method']
  if method == 'random':
    sampling = google.datalab.bigquery.Sampling.random(percent=args['percent'], count=count)
  elif method == 'hashed':
    sampling = google.datalab.bigquery.Sampling.hashed(field_name=args['field'],
                                            percent=args['percent'],
                                            count=count)
  elif method == 'sorted':
    ascending = args['order'] == 'ascending'
    sampling = google.datalab.bigquery.Sampling.sorted(args['field'],
                                            ascending=ascending,
                                            count=count)
  elif method == 'limit':
    sampling = google.datalab.bigquery.Sampling.default(count=count)
  else:
    sampling = google.datalab.bigquery.Sampling.default(count=count)

  if query:
    results = query.sample(sampling=sampling, dialect=args['dialect'],
                           billing_tier=args['billing'])
  elif view:
    results = view.sample(sampling=sampling)
  else:
    results = table.sample(sampling=sampling)
  if args['verbose']:
    print(results.sql)
  if args['profile']:
    return google.datalab.utils.commands.profile_df(results.to_dataframe())
  else:
    return results


def _create_cell(args, cell_body):
  """Implements the BigQuery cell magic used to create datasets and tables.

   The supported syntax is:

     %%bq create dataset -n|--name <name> [-f|--friendly <friendlyname>]
     [<description>]

   or:

     %%bq create table -n|--name <tablename> [--overwrite]
     [<YAML or JSON cell_body defining schema to use for tables>]

  Args:
    args: the argument following '%bq create <command>'.
  """
  if args['command'] == 'dataset':
    try:
      google.datalab.bigquery.Dataset(args['name']).create(friendly_name=args['friendly'],
                                                    description=cell_body)
    except Exception as e:
      print('Failed to create dataset %s: %s' % (args['name'], e))
  else:
    if cell_body is None:
      print('Failed to create %s: no schema specified' % args['name'])
    else:
      try:
        record = google.datalab.utils.commands.parse_config(cell_body, google.datalab.utils.commands.notebook_environment(), as_dict=False)
        schema = google.datalab.bigquery.Schema(record)
        google.datalab.bigquery.Table(args['name']).create(schema=schema, overwrite=args['overwrite'])
      except Exception as e:
        print('Failed to create table %s: %s' % (args['name'], e))


def _delete_cell(args, _):
  """Implements the BigQuery cell magic used to delete datasets and tables.

   The supported syntax is:

     %%bq delete dataset -n|--name <name>

   or:

     %%bq delete table -n|--name <name>

  Args:
    args: the argument following '%bq delete <command>'.
  """
  # TODO(gram): add support for wildchars and multiple arguments at some point. The latter is
  # easy, the former a bit more tricky if non-default projects are involved.
  if args['command'] == 'dataset':
    try:
      google.datalab.bigquery.Dataset(args['name']).delete()
    except Exception as e:
      print('Failed to delete dataset %s: %s' % (args['name'], e))
  else:
    try:
      google.datalab.bigquery.Table(args['name']).delete()
    except Exception as e:
      print('Failed to delete table %s: %s' % (args['name'], e))


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
  result = query.execute_dry_run(dialect=args['dialect'], billing_tier=args['billing'])
  return google.datalab.bigquery._query_stats.QueryStats(total_bytes=result['totalBytesProcessed'],
                                              is_cached=result['cacheHit'])


def _udf_cell(args, js):
  """Implements the Bigquery udf cell magic for ipython notebooks.

  The supported syntax is:
  %%bq udf --module <var>
  <js function>

  Args:
    args: the optional arguments following '%%bq udf'.
    js: the UDF declaration (inputs and outputs) and implementation in javascript.
  Returns:
    The results of executing the UDF converted to a dataframe if no variable
    was specified. None otherwise.
  """
  variable_name = args['module']
  if not variable_name:
    raise Exception('Declaration must be of the form %%bq udf --module <variable name>')

  # Parse out the input and output specification
  spec_pattern = r'\{\{([^}]+)\}\}'
  spec_part_pattern = r'[a-z_][a-z0-9_]*'

  specs = re.findall(spec_pattern, js)
  if len(specs) < 2:
    raise Exception('The JavaScript must declare the input row and output emitter parameters '
                    'using valid jsdoc format comments.\n'
                    'The input row param declaration must be typed as {{field:type, field2:type}} '
                    'and the output emitter param declaration must be typed as '
                    'function({{field:type, field2:type}}.')

  inputs = []
  input_spec_parts = re.findall(spec_part_pattern, specs[0], flags=re.IGNORECASE)
  if len(input_spec_parts) % 2 != 0:
    raise Exception('Invalid input row param declaration. The jsdoc type expression must '
                    'define an object with field and type pairs.')
  for n, t in zip(input_spec_parts[0::2], input_spec_parts[1::2]):
    inputs.append((n, t))

  outputs = []
  output_spec_parts = re.findall(spec_part_pattern, specs[1], flags=re.IGNORECASE)
  if len(output_spec_parts) % 2 != 0:
    raise Exception('Invalid output emitter param declaration. The jsdoc type expression must '
                    'define a function accepting an an object with field and type pairs.')
  for n, t in zip(output_spec_parts[0::2], output_spec_parts[1::2]):
    outputs.append((n, t))

  # Look for imports. We use a non-standard @import keyword; we could alternatively use @requires.
  # Object names can contain any characters except \r and \n.
  import_pattern = r'@import[\s]+(gs://[a-z\d][a-z\d_\.\-]*[a-z\d]/[^\n\r]+)'
  imports = re.findall(import_pattern, js)

  # Split the cell if necessary. We look for a 'function(' with no name and a header comment
  # block with @param and assume this is the primary function, up to a closing '}' at the start
  # of the line. The remaining cell content is used as support code.
  split_pattern = r'(.*)(/\*.*?@param.*?@param.*?\*/\w*\n\w*function\w*\(.*?^}\n?)(.*)'
  parts = re.match(split_pattern, js, re.MULTILINE | re.DOTALL)
  support_code = ''
  if parts:
    support_code = (parts.group(1) + parts.group(3)).strip()
    if len(support_code):
      js = parts.group(2)

  # Finally build the UDF object
  udf = google.datalab.bigquery.UDF(inputs, outputs, variable_name, js, support_code, imports)
  google.datalab.utils.commands.notebook_environment()[variable_name] = udf


def _execute_cell(args, cell_body):
  """Implements the BigQuery cell magic used to execute BQ queries.

   The supported syntax is:
   %%bq execute [-q|--sql <query identifier>] <other args>
   [<YAML or JSON cell_body or inline SQL>]

  Args:
    args: the arguments following '%bq execute'.
    cell_body: optional contents of the cell interpreted as YAML or JSON.
  Returns:
    The QueryResultsTable
  """
  query = _get_query_argument(args, cell_body, google.datalab.utils.commands.notebook_environment())
  if args['verbose']:
    print(query.sql)
  return query.execute(args['target'], table_mode=args['mode'], use_cache=not args['nocache'],
                       allow_large_results=args['large'], dialect=args['dialect'],
                       billing_tier=args['billing']).results


def _pipeline_cell(args, cell_body):
  """Implements the BigQuery cell magic used to validate, execute or deploy BQ pipelines.

   The supported syntax is:
   %%bq pipeline [-q|--sql <query identifier>] <other args> <action>
   [<YAML or JSON cell_body or inline SQL>]

  Args:
    args: the arguments following '%bq pipeline'.
    cell_body: optional contents of the cell interpreted as YAML or JSON.
  Returns:
    The QueryResultsTable
  """
  if args['action'] == 'deploy':
    raise Exception('Deploying a pipeline is not yet supported')

  env = {}
  for key, value in google.datalab.utils.commands.notebook_environment().items():
    if isinstance(value, google.datalab.bigquery._udf.UDF):
      env[key] = value

  query = _get_query_argument(args, cell_body, env)
  if args['verbose']:
    print(query.sql)
  if args['action'] == 'dryrun':
    print(query.sql)
    result = query.execute_dry_run()
    return google.datalab.bigquery._query_stats.QueryStats(total_bytes=result['totalBytesProcessed'],
                                                is_cached=result['cacheHit'])
  if args['action'] == 'run':
    return query.execute(args['target'], table_mode=args['mode'], use_cache=not args['nocache'],
                         allow_large_results=args['large'], dialect=args['dialect'],
                         billing_tier=args['billing']).results


def _table_line(args):
  """Implements the BigQuery table magic used to display tables.

   The supported syntax is:
   %bq table -t|--table <name> <other args>

  Args:
    args: the arguments following '%bq table'.
  Returns:
    The HTML rendering for the table.
  """
  # TODO(gram): It would be good to turn _table_viewer into a class that has a registered
  # renderer. That would allow this to return a table viewer object which is easier to test.
  name = args['table']
  table = _get_table(name)
  if table and table.exists():
    fields = args['cols'].split(',') if args['cols'] else None
    html = _table_viewer(table, rows_per_page=args['rows'], fields=fields)
    return IPython.core.display.HTML(html)
  else:
    raise Exception('Table %s does not exist; cannot display' % name)


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


def _schema_line(args):
  """Implements the BigQuery schema magic used to display table/view schemas.

  Args:
    args: the arguments following '%bq schema'.
  Returns:
    The HTML rendering for the schema.
  """
  # TODO(gram): surely we could just return the schema itself?
  name = args['table'] if args['table'] else args['view']
  if name is None:
    raise Exception('No table or view specified; cannot show schema')

  schema = _get_schema(name)
  if schema:
    html = _repr_html_table_schema(schema)
    return IPython.core.display.HTML(html)
  else:
    raise Exception('%s is not a schema and does not appear to have a schema member' % name)


def _render_table(data, fields=None):
  """ Helper to render a list of dictionaries as an HTML display object. """
  return IPython.core.display.HTML(google.datalab.utils.commands.HtmlBuilder.render_table(data, fields))


def _render_list(data):
  """ Helper to render a list of objects as an HTML list object. """
  return IPython.core.display.HTML(google.datalab.utils.commands.HtmlBuilder.render_list(data))


def _datasets_line(args):
  """Implements the BigQuery datasets magic used to display datasets in a project.

   The supported syntax is:

       %bq datasets [-f <filter>] [-p|--project <project_id>]

  Args:
    args: the arguments following '%bq datasets'.
  Returns:
    The HTML rendering for the table of datasets.
  """
  filter_ = args['filter'] if args['filter'] else '*'
  return _render_list([str(dataset) for dataset in google.datalab.bigquery.Datasets(args['project'])
                       if fnmatch.fnmatch(str(dataset), filter_)])


def _tables_line(args):
  """Implements the BigQuery tables magic used to display tables in a dataset.

   The supported syntax is:

       %bq tables -p|--project <project_id>  -d|--dataset <dataset_id>

  Args:
    args: the arguments following '%bq tables'.
  Returns:
    The HTML rendering for the list of tables.
  """
  filter_ = args['filter'] if args['filter'] else '*'
  if args['dataset']:
    if args['project'] is None:
      datasets = [google.datalab.bigquery.Dataset(args['dataset'])]
    else:
      datasets = [google.datalab.bigquery.Dataset((args['project'], args['dataset']))]
  else:
    datasets = google.datalab.bigquery.Datasets(args['project'])

  tables = []
  for dataset in datasets:
    tables.extend([str(table) for table in dataset if fnmatch.fnmatch(str(table), filter_)])

  return _render_list(tables)


def _extract_line(args):
  """Implements the BigQuery extract magic used to extract table data to GCS.

   The supported syntax is:

       %bq extract -S|--source <table> -D|--destination <url> <other_args>

  Args:
    args: the arguments following '%bq extract'.
  Returns:
    A message about whether the extract succeeded or failed.
  """
  name = args['source']
  source = google.datalab.utils.commands.get_notebook_item(name)
  if not source:
    source = _get_table(name)

  if not source:
    raise Exception('No source named %s found' % name)
  elif isinstance(source, google.datalab.bigquery.Table) and not source.exists():
    raise Exception('Table %s does not exist' % name)
  else:

    job = source.extract(args['destination'],
                         format='CSV' if args['format'] == 'csv' else 'NEWLINE_DELIMITED_JSON',
                         compress=args['compress'],
                         csv_delimiter=args['delimiter'],
                         csv_header=args['header'])
    if job.failed:
      raise Exception('Extract failed: %s' % str(job.fatal_error))
    elif job.errors:
      raise Exception('Extract completed with errors: %s' % str(job.errors))


def _load_cell(args, schema):
  """Implements the BigQuery load magic used to load data from GCS to a table.

   The supported syntax is:

       %bq load -S|--source <source> -D|--destination <table>  <other_args>

  Args:
    args: the arguments following '%bq load'.
    schema: a JSON schema for the destination table.
  Returns:
    A message about whether the load succeeded or failed.
  """
  name = args['destination']
  table = _get_table(name)
  if not table:
    table = google.datalab.bigquery.Table(name)

  if table.exists():
    if args['mode'] == 'create':
      raise Exception('%s already exists; use --append or --overwrite' % name)
  elif schema:
    table.create(json.loads(schema))
  elif not args['infer']:
    raise Exception(
        'Table does not exist, no schema specified in cell and no --infer flag; cannot load')

  # TODO(gram): we should probably try do the schema infer ourselves as BQ doesn't really seem
  # to be able to do it. Alternatively we can drop the --infer argument and force the user
  # to use a pre-existing table or supply a JSON schema.
  csv_options = google.datalab.bigquery.CSVOptions(delimiter=args['delimiter'],
                                        skip_leading_rows=args['skip'],
                                        allow_jagged_rows=not args['strict'],
                                        quote=args['quote'])
  job = table.load(args['source'],
                   mode=args['mode'],
                   source_format=('CSV' if args['format'] == 'csv' else 'NEWLINE_DELIMITED_JSON'),
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

  # %%bq sample
  _add_command(parser, _create_sample_subparser, _sample_cell)

  # %%bq create
  _add_command(parser, _create_create_subparser, _create_cell)

  # %%bq delete
  _add_command(parser, _create_delete_subparser, _delete_cell)

  # %%bq dryrun
  _add_command(parser, _create_dry_run_subparser, _dryrun_cell)

  # %%bq udf
  _add_command(parser, _create_udf_subparser, _udf_cell, cell_required=True)

  # %%bq execute
  _add_command(parser, _create_execute_subparser, _execute_cell)

  # %%bq pipeline
  _add_command(parser, _create_pipeline_subparser, _pipeline_cell)

  # %bq table
  _add_command(parser, _create_table_subparser, _table_line, cell_prohibited=True)

  # %bq schema
  _add_command(parser, _create_schema_subparser, _schema_line, cell_prohibited=True)

  # %bq datasets
  _add_command(parser, _create_datasets_subparser, _datasets_line, cell_prohibited=True)

  # %bq tables
  _add_command(parser, _create_tables_subparser, _tables_line, cell_prohibited=True)

  # %bq extract
  _add_command(parser, _create_extract_subparser, _extract_line, cell_prohibited=True)

  # %bq load
  # TODO(gram): need some additional help, esp. around the option of specifying schema in
  # cell body and how schema infer may fail.
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
            google.datalab: 'nbextensions/gcpdatalab'
          }}
        }},
        shim: {{
          plotly: {{
            deps: ['d3', 'jquery'],
            exports: 'plotly'
          }}
        }}
      }});

      require(['google.datalab/charting', 'google.datalab/element!{div_id}', 'base/js/events',
          'google.datalab/style!/nbextensions/gcpdatalab/charting.css'],
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
            google.datalab: 'nbextensions/gcpdatalab'
          }
        },
      });

      require(['google.datalab/bigquery', 'google.datalab/element!%s',
          'google.datalab/style!/nbextensions/gcpdatalab/bigquery.css'],
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
