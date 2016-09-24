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

try:
  import IPython
  import IPython.core.magic
except ImportError:
  raise Exception('This module can only be loaded in ipython.')


import collections
import datetime
import fnmatch
import google.cloud.ml
import json
import math
import os
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import urllib
import yaml

import datalab.context
import datalab.data
import datalab.ml
import datalab.utils.commands


@IPython.core.magic.register_line_cell_magic
def ml(line, cell=None):
  """Implements the ml line cell magic.

  Args:
    line: the contents of the ml line.
    cell: the contents of the ml cell.
  Returns:
    The results of executing the cell.
  """
  parser = datalab.utils.commands.CommandParser(prog='ml', description="""
Execute various ml-related operations. Use "%%ml <command> -h" for help on a specific command.
""")
  train_parser = parser.subcommand('train', 'Run a training job.')
  train_parser.add_argument('--cloud',
                            help='Whether to run the training job in the cloud.',
                            action='store_true', default=False)
  train_parser.set_defaults(func=_train)
  jobs_parser = parser.subcommand('jobs', 'List jobs in a project.')
  jobs_parser.add_argument('--count',
                           help='The number of jobs to browse from head, default to 10.')
  jobs_parser.add_argument('--filter', help='Filter on jobs.')
  jobs_parser.add_argument('--name',
                           help='The name of the operation to retrieve. If provided, show ' +
                                'detailed information of the operation')
  jobs_parser.add_argument('--trials',
                            help='Whether to show hyperparams tuning graph.',
                            action='store_true', default=False)
  jobs_parser.set_defaults(func=_jobs)
  summary_parser = parser.subcommand('summary', 'List or view summary events.')
  summary_parser.add_argument('--dir',
                              help='A list of dirs to look for events. Can be local or GCS path.',
                              nargs='+', required=True)
  summary_parser.add_argument('--name',
                              help='Names of the summary event. If provided, ' +
                                   'plot specified events in same graph (so make sure their ' +
                                   'units match). Otherwise, list all the unique event ' +
                                   'names from files in the directory.', nargs='*')
  summary_parser.add_argument('--time', help='Whether to plot time events only.',
                              action='store_true', default=False)
  summary_parser.add_argument('--step', help='Whether to plot step events only.',
                              action='store_true', default=False)
  summary_parser.set_defaults(func=_summary)
  features_parser = parser.subcommand('features', 'Generate featureset class template.')
  features_parser.set_defaults(func=_features)
  predict_parser = parser.subcommand('predict', 'Get prediction results given data instances.')
  predict_parser.add_argument('--cloud',
                              help='Whether to run the prediction in the cloud.',
                              action='store_true', default=False)
  predict_parser.add_argument('--model',
                              help='Model identifier. In local prediction, it is the path to ' +
                                   'a model directory. In cloud prediction (--cloud), it is ' +
                                   'model.version.', required=True)
  predict_parser.add_argument('--label',
                              help='In classification scenario, which output in the graph ' +
                                   'is the label index. If provided, the index will be ' +
                                   'converted to label string.')
  predict_parser.add_argument('--data',
                              help='The instance data used to predict. It can be a dataframe ' +
                                   'or a list defined in another cell. If not provided, the ' +
                                   'data needs to be provided in current cell input.')
  predict_parser.add_argument('--project',
                              help='The project for the cloud model to use. if not provided, ' +
                                   'current project is used. Only needed in cloud prediction ' +
                                   '(--cloud)')
  predict_parser.set_defaults(func=_predict)
  model_parser = parser.subcommand('model', 'List or view models.')
  model_parser.add_argument('--name',
                            help='The name and version of the model. If "model.version", ' +
                                 'display the details of the model version. If "model", ' +
                                 'list the versions of the model. If not provided, list ' +
                                 'models under the project.')
  model_parser.add_argument('--project',
                            help='The project under which it looks for models. if not ' +
                                 'provided, current project is used.')
  model_parser.set_defaults(func=_model)
  deploy_parser = parser.subcommand('deploy', 'Deploy a model.')
  deploy_parser.add_argument('--name',
                             help='The name and version of the model in the form of ' +
                                  '"model.version".', required=True)
  deploy_parser.add_argument('--path',
                             help='The Google Cloud Storage path of the directory that ' +
                                  'contains an exported model.', required=True)
  deploy_parser.add_argument('--project',
                             help='The project under which the model will be deployed. if not ' +
                                  'provided, current project is used.')
  deploy_parser.set_defaults(func=_deploy)
  delete_parser = parser.subcommand('delete', 'Delete a model or a model version.')
  delete_parser.add_argument('--name',
                             help='The name and version of the model. If "model.version", ' +
                                  'delete the model version. If "model", delete the model.',
                             required=True)
  delete_parser.add_argument('--project',
                             help='The project under which the model or version will be ' +
                                  'deleted. if not provided, current project is used.')
  delete_parser.set_defaults(func=_delete)
  preprocess_parser = parser.subcommand('preprocess', 'Generate preprocess code template.')
  preprocess_parser.add_argument('--cloud',
                                 help='Whether to produce code running in cloud.',
                                 action='store_true', default=False)
  preprocess_parser.set_defaults(func=_preprocess)
  evaluate_parser = parser.subcommand('evaluate', 'Generate evaluate code template.')
  evaluate_parser.add_argument('--cloud',
                               help='Whether to produce code running in cloud.',
                               action='store_true', default=False)
  evaluate_parser.set_defaults(func=_evaluate)
  dataset_parser = parser.subcommand('dataset', 'Define dataset to explore data.')
  dataset_parser.add_argument('--name',
                              help='The name of the dataset to define.', required=True)
  dataset_parser.set_defaults(func=_dataset)
  module_parser = parser.subcommand('module', 'Define a trainer module.')
  module_parser.add_argument('--name', help='The name of the module.', required=True)
  module_parser.add_argument('--main',
                             help='Whether this module is has main function in the trainer ' +
                                  'package.',
                             action='store_true', default=False)
  module_parser.set_defaults(func=_module)
  package_parser = parser.subcommand('package','Create a trainer package from all modules ' +
                                     'defined with %%ml module.')
  package_parser.add_argument('--name', help='The name of the package.', required=True)
  package_parser.add_argument('--output', help='the output dir of the package.', required=True)
  package_parser.set_defaults(func=_package)
  namespace = datalab.utils.commands.notebook_environment()
  return datalab.utils.commands.handle_magic_line(line, cell, parser, namespace=namespace)


def _get_replica_count(config):
  worker_count = config.get('worker_count', 0)
  parameter_server_count = config.get('parameter_server_count', 0)
  if worker_count > 0 or parameter_server_count > 0:
    return 1, worker_count, parameter_server_count
  scale_tier = config.get('scale_tier', 'BASIC')
  if scale_tier == 'BASIC':
    return 1, 0, 0
  else:
    return 1, 1, 1


def _local_train_callback(replica_spec, new_msgs, done, all_msgs):
  if new_msgs:
    all_msgs += new_msgs
    del all_msgs[0:-20]
    IPython.display.clear_output()
    IPython.display.display_html('<p>Job Running...</p>', raw=True)
    log_file_html = ''
    log_url_prefix = ''
    if datalab.context._utils._in_datalab_docker():
      log_url_prefix = '/_nocachecontent/'
    for job_type, replicas in replica_spec.iteritems():
      if replicas > 0:
        log_file_html += ('<a href="%s" target="_blank">%s log</a>&nbsp;&nbsp;'
                          % (log_url_prefix + job_type, job_type))
    IPython.display.display_html(log_file_html, raw=True)
    IPython.display.display_html('<br/>'.join(all_msgs), raw=True)
  if done:
    IPython.display.display_html('<p>Job Finished.</p>', raw=True)


def _output_train_template():
  content = """%%ml train [--cloud]
package_uris: gs://your-bucket/my-training-package.tar.gz
python_module: your_program.your_module
scale_tier: BASIC
region: us-central1
args:
  string_arg: value
  int_arg: value
  appendable_arg:
    - value1
    - value2
"""
  IPython.get_ipython().set_next_input(content)
  parameters = ['package_uris', 'python_module', 'scale_tier', 'region', 'args']
  required_local = [False, False, False, False, False]
  required_cloud = [True, True, True, True, False]
  description = [
    'A GCS or local (for local run only) path to your python training program package.',
    'The module to run.',
    'Type of resources requested for the job. On local run, BASIC means 1 master process only, ' +
      'and any other values mean 1 master 1 worker and 1 ps processes. But you can also ' +
      'override the values by setting worker_count and parameter_server_count. ' +
      'On cloud, see service definition for possible values.',
    'Where the training job runs. For cloud run only.',
    'Args that will be passed to your training program.'
  ]
  data = [{'Parameters': x[0], 'Local Run Required': str(x[1]),
           'Cloud Run Required': str(x[2]), 'Description': x[3]}
          for x in zip(parameters, required_local, required_cloud, description)]
  html = ('<p>A training input template is created in next cell for you. See cell input ' +
          'instructions below.</p>')
  html += datalab.utils.commands.HtmlBuilder.render_table(data,
      ['Parameters', 'Local Run Required', 'Cloud Run Required', 'Description'])

  return IPython.core.display.HTML(html)


def _train(args, cell):
  """ Train a model. """
  if not cell:
    return _output_train_template()

  env = datalab.utils.commands.notebook_environment()
  config = datalab.utils.commands.parse_config(cell, env)
  if args['cloud']:
    datalab.utils.commands.validate_config_must_have(config,
        ['package_uris', 'python_module', 'scale_tier', 'region'])
    runner = datalab.ml.CloudRunner(config)
    job_info = runner.run()
    job_short_name = job_info['jobId']
    html = '<p>Job "%s" was submitted successfully.<br/>' % job_short_name
    html += 'Run "%%ml jobs --name %s" to view the status of the job.</p>' % job_short_name
    log_url_query_strings = {
      'project': datalab.context.Context.default().project_id,
      'resource': 'ml.googleapis.com/job_id/' + job_short_name
    }
    log_url = 'https://console.developers.google.com/logs/viewer?' + \
        urllib.urlencode(log_url_query_strings)
    html += '<p>Click <a href="%s" target="_blank">here</a> to view cloud log. <br/>' % log_url
    html += 'Start TensorBoard by running "%tensorboard start --logdir=&lt;YourLogDir&gt;".</p>'
    return IPython.core.display.HTML(html);
  else:
    # local training
    package_path = None
    if 'package_uris' not in config:
      if '_ml_modules_' not in env:
        raise Exception('Expect either modules defined with "%%ml module", ' +
                        'or "package_uris" in cell.')
      if '_ml_modules_main_' not in env:
        raise Exception('Expect one ml module defined with "--main flag" as the python ' +
                        'program entry point.')
      package_path = datalab.ml.Packager().package(env['_ml_modules_'], 'trainer')
      config['package_uris'] = package_path
      config['python_module'] = 'trainer.' + env['_ml_modules_main_']

    trainer_uri = config['package_uris']
    module_name = config['python_module']
    masters, workers, parameter_servers = _get_replica_count(config)
    replica_spec = {'master': masters, 'worker': workers, 'ps': parameter_servers}
    all_messages = []
    log_dir = os.getcwd()
    if datalab.context._utils._in_datalab_docker():
      log_dir = '/datalab/nocachecontent'
      if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    program_args = config.get('args', None)
    runner = datalab.ml.LocalRunner(trainer_uri, module_name, log_dir, replica_spec, program_args, config)
    runner.run(_local_train_callback, all_messages, 3)
    if package_path is not None:
      os.remove(package_path)


def _plot_hyperparams_tuning(training_input, training_output):
  if ('hyperparameters' not in training_input or 'trials' not in training_output or
      len(training_output['trials']) == 0):
    print 'No trials found. Maybe none of the trials has completed'
    return

  maximize = training_input['hyperparameters']['goal'] == 'MAXIMIZE'
  hyperparam_scales = {}
  for param in training_input['hyperparameters']['params']:
    hyperparam_scales[param['parameterName']] = param.get('scaleType', '')
  instances = []
  for trial in training_output['trials']:
    if 'finalMetric' not in trial:
      continue
    instance = collections.OrderedDict()
    instance.update({'Objective': trial['finalMetric']['objectiveValue']})
    instance.update({'Trial': trial['trialId']})
    instance.update({'Training Step': trial['finalMetric']['trainingStep']})
    hyperparams = dict(trial['hyperparameters'])
    for k in trial['hyperparameters'].keys():
      if hyperparam_scales.get(k, '') == 'UNIT_LOG_SCALE':
        hyperparams[k + '(log)'] = math.log10(float(hyperparams[k]))
    instance.update(hyperparams)
    instances.append(instance)
  if len(instances) == 0:
    print 'No finalMetric found in any trials. '
    return

  instances_sorted = sorted(instances, key=lambda k: k['Objective'], reverse=maximize)
  # Convert list of dictionary to dictionary of list so it is more compact.
  data = instances_sorted[0]
  for k in data.keys():
    data[k] = [d[k] for d in instances_sorted]

  HTML_TEMPLATE = """
<div id="%s" class="parcoords" style="height:300px;"></div>
<div id="%s" class="parcoords_grid"></div>
<script>
  require.config({
    paths: {
      d3: 'http://d3js.org/d3.v3.min',
      sylvester: '/nbextensions/gcpdatalab/extern/sylvester',
      parcoords: '/nbextensions/gcpdatalab/extern/d3.parcoords',
    },
    shim: {
      parcoords: {
        deps: ['d3', 'sylvester'],
        exports: 'd3'
      },
    }
  });
  require(['parcoords',
           '/nbextensions/gcpdatalab/parcoords.js',
           'nbextensions/gcpdatalab/style!/nbextensions/gcpdatalab/extern/d3.parcoords.css'],
          function(d3, lib) {
            var data = %s;
            lib.plot(d3, %s, %s, data, '%s', '%s');
          });
</script>
"""
  graph_id = 'v' + datalab.utils.commands.Html.next_id()
  grid_id = 'g' + datalab.utils.commands.Html.next_id()
  color_range_string = json.dumps([min(data['Objective']), max(data['Objective'])])
  data_string = json.dumps(data)
  maximize_string = json.dumps(maximize)
  html = HTML_TEMPLATE % (graph_id, grid_id, data_string, color_range_string, maximize_string, graph_id, grid_id)
  return IPython.core.display.HTML(html)


def _jobs(args, _):
  """ List the ML jobs in a project. """
  jobs = datalab.ml.Jobs(filter=args['filter'])
  if args['name']:
    job = jobs.get_job_by_name(args['name'])
    if args['trials']:
      if ('trainingInput' not in job.info or 'trainingOutput' not in job.info):
        print 'job %s doesn\'t seem like a hyperparameter tuning training job.' % args['name']
        return
      return _plot_hyperparams_tuning(job.info['trainingInput'], job.info['trainingOutput'])
    else:
      job_yaml = yaml.safe_dump(job.info)
    return datalab.utils.commands.render_text(job_yaml, preformatted=True)
  else:
    count = int(args['count'] or 10)
    data = []
    for job in jobs:
      if count <= 0:
        break
      count -= 1
      data.append({'Id': job['jobId'], 'State': job.get('state', 'UNKNOWN')})
    return datalab.utils.commands.render_dictionary(data, ['Id', 'State'])


def _plot(data, x_name, x_title, y_names):
  y_title = ','.join(y_names)
  layout = go.Layout(
    title=y_title,
    xaxis=dict(
      title=x_title,
    ),
    yaxis=dict(
      title=y_title,
    )
  )
  plot_data = []
  for trace_name, trace_data in data.iteritems():
    for y_name, events in zip(y_names, trace_data):
      x = [d[x_name] for d in events]
      y = [d[y_name] for d in events]
      plot_data.append({'x': x, 'y': y, 'name': y_name + '-' + trace_name})
  fig = go.Figure(data=plot_data, layout=layout)
  iplot(fig)


def get_dirs(pattern):
  # given a dir path, find all matching dirs
  # for example:
  #  input: gs://mybucket/iris/hp/*/summaries
  #  output: [gs://mybucket/iris/hp/1/summaries, gs://mybucket/iris/hp/2/summaries]
  dirs = set()
  path = pattern.rstrip('/')
  for p in google.cloud.ml.util._file.glob_files(path + '/*'):
    dir = None
    while True:
      p = p[:p.rfind('/')]
      if fnmatch.fnmatch(p, path):
        dir = p
      else:
        break
    if dir:
      dirs.add(dir)
  return list(dirs)


def _summary(args, _):
  """ Display summary events in a directory. """
  dirs = args['dir']
  event_names = args['name']
  if event_names is not None and len(event_names) > 0:
    time_data = {}
    step_data = {}
    dir_index = 0
    for dir_pattern in dirs:
      for dir in get_dirs(dir_pattern):
        dir_index += 1
        summary = datalab.ml.Summary(dir)
        trace_events_time = []
        trace_events_step = []
        for event_name in event_names:
          events_time, events_step = summary.get_events(event_name)
          for e in events_time:
            e['time'] = e['time'].total_seconds()
          trace_events_time.append(events_time)
          trace_events_step.append(events_step)
        # Try to find 'label' file under the dir. If found, use the content as label.
        # Otherwise, use dir name as label.
        label = dir
        label_file = os.path.join(dir, 'label')
        if google.cloud.ml.util._file.file_exists(label_file) == True:
          label = 'dir%d/' % dir_index + google.cloud.ml.util._file.load_file(label_file)
        time_data[label] = trace_events_time
        step_data[label] = trace_events_step
    if (not args['time'] and not args['step']) or args['time']:
      _plot(time_data, 'time', 'seconds', event_names)
    if (not args['time'] and not args['step']) or args['step']:
      _plot(step_data, 'step', 'step', event_names)
  else:
    event_names = []
    for dir_pattern in dirs:
      for dir in get_dirs(dir_pattern):
        summary = datalab.ml.Summary(dir)
        event_names += summary.list_events()
    event_names = list(set(event_names))  # remove duplicates
    return datalab.utils.commands.render_list(event_names)


def _output_features_template():
  content = """%%ml features
path: REQUIRED_Fill_In_Gcs_or_Local_Path
headers: List_Of_Column_Names_Seperated_By_Comma
target: REQUIRED_Fill_In_Name_Or_Index_Of_Target_Column
id: Fill_In_Name_Or_Index_Of_Id_Column
format: csv_or_tsv
"""
  IPython.get_ipython().set_next_input(content, replace=True)


def _features(args, cell):
  """ Generate FeatureSet Class From Data"""
  if not cell:
    _output_features_template()
    return

  env = datalab.utils.commands.notebook_environment()
  config = datalab.utils.commands.parse_config(cell, env)
  datalab.utils.commands.validate_config(config, ['path', 'target'],
      optional_keys=['headers', 'id', 'format'])
  format = config.get('format', 'csv')
  # For now, support CSV and TSV only.
  datalab.utils.commands.validate_config_value(format, ['csv', 'tsv'])
  delimiter = ',' if format == 'csv' else '\t'
  csv = datalab.data.Csv(config['path'], delimiter=delimiter)
  headers = None
  if 'headers' in config:
    headers = [e.strip() for e in config['headers'].split(',')]
  df = csv.browse(max_lines=100, headers=headers)
  command = '%ml features\n' + cell
  _output_featureset_template(df.dtypes, config['target'], config.get('id', None), command)


def _output_featureset_template(dtypes, target_column, id_column, command):
  if target_column not in dtypes:
    if type(target_column) is int:
      target_column = dtypes.keys()[target_column]
    else:
      raise Exception('Column "%s" not found. It can be a name in headers, or an index number.'
                      % target_column)
  if id_column is not None and id_column not in dtypes:
    if type(id_column) is int:
      id_column = dtypes.keys()[id_column]
    else:
      raise Exception('Column "%s" not found. It can be a name in headers, or an index number.'
                      % id_column)
  is_regression = str(dtypes[target_column]).startswith('int') or \
      str(dtypes[target_column]).startswith('float')
  scenario = 'regression' if is_regression == True else 'classification'
  columns_remaining = dict(dtypes)
  command_lines = command.split('\n')
  # add spaces in the beginning so they are aligned with others.
  command_formatted = '\n'.join(['        ' + line for line in command_lines])
  content = """import google.cloud.ml.features as features

class CsvFeatures(object):
  \"\"\"This class is generated from command line:
%s
        Please modify it as appropriate!!!
  \"\"\"
  csv_columns = (%s)
  %s = features.target('%s').%s()
""" % (command_formatted, ','.join(["'" + e + "'" for e in dtypes.keys()]),
       target_column.replace('-', '_'), target_column, scenario)
  del columns_remaining[target_column]

  if id_column is not None:
    content += """  %s = features.key('%s')\n""" % (id_column.replace('-', '_'), id_column)
    del columns_remaining[id_column]

  text_columns = [k for k,v in columns_remaining.iteritems() if str(v) == 'object']
  categorical_columns = [k for k,v in columns_remaining.iteritems() if str(v) == 'category']
  numeric_columns = [k for k in columns_remaining.keys() if k not in text_columns and
                     k not in categorical_columns]
  if len(numeric_columns) + len(categorical_columns) > 0:
    content += """  attrs = [\n"""
    for numeric_name in numeric_columns:
      content += """      features.numeric('%s').identity(),\n""" % numeric_name
    for categorical_name in categorical_columns:
      content += """      features.categorical('%s'),\n""" % categorical_name
    content += """  ]\n"""
  if len(text_columns) > 0:
    for text_name in text_columns:
      content += """  %s = features.text('%s').bag_of_words(vocab_size=10000)\n\n""" % \
          (text_name.replace('-', '_'), text_name)
  IPython.get_ipython().set_next_input(content, replace=True)


def _predict(args, cell):
  if args['data'] is not None:
    instances = datalab.utils.commands.get_notebook_item(args['data'])
    if instances is None:
      raise Exception('Data "%s" is not defined' % args['data'])
  elif cell is not None:
    instances = []
    lines = cell.split('\n')
    for line in lines:
      instances.append(line)
  else:
    raise Exception('Expect instance data. Can be provided in input cell, or through '
                    '--data args.')
  if args['cloud']:
    parts = args['model'].split('.')
    if len(parts) != 2:
      raise Exception('Invalid model name for cloud prediction. Use "model.version".')
    lp = datalab.ml.CloudPredictor(parts[0], parts[1],
                                   label_output=args['label'],
                                   project_id=args['project'])
  else:
    lp = datalab.ml.LocalPredictor(args['model'],
                                   label_output=args['label'])
  return datalab.utils.commands.render_text(yaml.safe_dump(lp.predict(instances),
                                                           default_flow_style=False),
                                            preformatted=True)


def _model(args, _):
  if args['name'] is None:
    data = list(datalab.ml.CloudModels(project_id=args['project']))
    if len(data) > 0:
      return datalab.utils.commands.render_dictionary(data, data[0].keys())
    print 'No models found.'
    return

  parts = args['name'].split('.')
  if len(parts) == 1:
    data = list(datalab.ml.CloudModelVersions(parts[0], project_id=args['project']))
    if len(data) > 0:
      return datalab.utils.commands.render_dictionary(data, data[0].keys())
    print 'No versions found.'
    return
  elif len(parts) == 2:
    versions = datalab.ml.CloudModelVersions(parts[0], project_id=args['project'])
    version_yaml = yaml.safe_dump(versions.get(parts[1]))
    return datalab.utils.commands.render_text(version_yaml, preformatted=True)
  else:
    raise Exception('Too many "." in name. Use "model" or "model.version".')


def _deploy(args, _):
  parts = args['name'].split('.')
  if len(parts) != 2:
    raise Exception('Invalid model name. Use "model.version".')
  versions = datalab.ml.CloudModelVersions(parts[0], project_id=args['project'])
  versions.deploy(parts[1], args['path'])


def _delete(args, _):
  parts = args['name'].split('.')
  if len(parts) == 1:
    models = datalab.ml.CloudModels(project_id=args['project'])
    models.delete(parts[0])
  elif len(parts) == 2:
    versions = datalab.ml.CloudModelVersions(parts[0], project_id=args['project'])
    versions.delete(parts[1])
  else:
    raise Exception('Too many "." in name. Use "model" or "model.version".')


def _output_preprocess_template(is_cloud):
  content = """%%ml preprocess"""
  if is_cloud:
    content += ' --cloud'
  content += """
train_data_path: REQUIRED_Fill_In_Training_Data_Path
eval_data_path: Fill_In_Evaluation_Data_Path
data_format: REQUIRED_CSV_or_JSON
output_dir: REQUIRED_Fill_In_Output_Path
feature_set_class_name: REQUIRED_Fill_In_FeatureSet_Class_name
"""
  IPython.get_ipython().set_next_input(content, replace=True)


def _pipeline_definition_code(is_cloud, job_name_prefix):
  # TODO: Remove 'extra_packages' once it is not needed by dataflow.
  if is_cloud:
    content_pipeline = \
"""import datetime
options = {
    'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
    'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
    'job_name': '%s' + '-' + datetime.datetime.now().strftime('%%y%%m%%d-%%H%%M%%S'),
    'project': '%s',
    'extra_packages': ['gs://cloud-ml/sdk/cloudml-0.1.2.latest.tar.gz'],
    'teardown_policy': 'TEARDOWN_ALWAYS',
    'no_save_main_session': True
}
opts = beam.pipeline.PipelineOptions(flags=[], **options)
pipeline = beam.Pipeline('DataflowPipelineRunner', options=opts)
""" % (job_name_prefix, datalab.context.Context.default().project_id)
  else:
    content_pipeline = """pipeline = beam.Pipeline('DirectPipelineRunner')\n"""

  return content_pipeline


def _header_code(command):
  header = \
"""\"\"\"
Following code is generated from command line:
%s\n
Please modify as appropriate!!!
\"\"\"
""" % command
  return header


def _output_preprocess_code_template(command, is_cloud, data_format, train_data_path,
                                     output_dir, feature_set_class_name, eval_data_path=None):
  content_header = _header_code(command)

  content_imports = \
"""import apache_beam as beam
import google.cloud.ml as ml
import google.cloud.ml.dataflow.io.tfrecordio as tfrecordio
import google.cloud.ml.io as io
import os
"""

  if data_format == 'CSV':
    coder = """io.CsvCoder.from_feature_set(feature_set, feature_set.csv_columns)"""
  else: # JSON
    coder = """io.JsonCoder.from_feature_set(feature_set)"""
  job_name_prefix = 'preprocess-' + feature_set_class_name.lower().replace('_', '-')

  content_defines = \
"""feature_set = %s()
OUTPUT_DIR = '%s'
%s
""" % (feature_set_class_name, output_dir, _pipeline_definition_code(is_cloud, job_name_prefix))

  content_preprocessing = \
"""training_data = beam.io.TextFileSource(
    '%s',
    strip_trailing_newlines=True,
    coder=%s)
train = pipeline | beam.Read('ReadTrainingData', training_data)
""" % (train_data_path, coder)
  if eval_data_path is not None:
    content_preprocessing += \
"""eval_data = beam.io.TextFileSource(
    '%s',
    strip_trailing_newlines=True,
    coder=%s)
eval = pipeline | beam.Read('ReadEvalData', eval_data)
""" % (eval_data_path, coder)

  eval_features = ', eval_features' if eval_data_path is not None else ''
  eval_input = ', eval' if eval_data_path is not None else ''
  content_preprocessing += \
"""(metadata, train_features%s) = ((train%s) |
    ml.Preprocess('Preprocess', feature_set, input_format='csv',
                  format_metadata={'headers': feature_set.csv_columns}))
train_parameters = tfrecordio.TFRecordParameters(
    file_path_prefix=os.path.join(OUTPUT_DIR, 'features_train'),
    file_name_suffix='',
    shard_file=False,
    compress_file=True)
""" % (eval_features, eval_input)
  if eval_data_path is not None:
    content_preprocessing += \
"""eval_parameters = tfrecordio.TFRecordParameters(
    file_path_prefix=os.path.join(OUTPUT_DIR, 'features_eval'),
    file_name_suffix='',
    shard_file=False,
    compress_file=True)
"""
  eval_output = ', eval_features' if eval_data_path is not None else ''
  eval_parameters = ', eval_parameters' if eval_data_path is not None else ''
  content_preprocessing += \
"""(metadata, train_features%s) | (
    io.SavePreprocessed('SavingData', OUTPUT_DIR,
                        file_parameters_list=[
                            os.path.join(OUTPUT_DIR, 'metadata.yaml'),
                            train_parameters%s]))
""" % (eval_output, eval_parameters)

  content_run = """pipeline.run()"""

  content = \
"""
# header
%s
# imports
%s
# defines
%s
# preprocessing
%s
# run pipeline
%s
""" % (content_header, content_imports, content_defines, content_preprocessing, content_run)
  IPython.get_ipython().set_next_input(content, replace=True)


def _preprocess(args, cell):
  if not cell:
    _output_preprocess_template(args['cloud'])
    return

  env = datalab.utils.commands.notebook_environment()
  config = datalab.utils.commands.parse_config(cell, env)
  datalab.utils.commands.validate_config(config,
     ['train_data_path', 'data_format', 'output_dir', 'feature_set_class_name'],
     optional_keys=['eval_data_path'])
  datalab.utils.commands.validate_config_value(config['data_format'], ['CSV', 'JSON'])
  command = '%%ml preprocess'
  if args['cloud']:
    command += ' --cloud'
  command += '\n' + cell
  _output_preprocess_code_template(command, args['cloud'], config['data_format'],
      config['train_data_path'], config['output_dir'], config['feature_set_class_name'],
      eval_data_path=config.get('eval_data_path', None))


def _output_evaluate_template(is_cloud):
  content = """%%ml evaluate"""
  if is_cloud:
    content += ' --cloud'
  content += """
preprocessed_eval_data_path: REQUIRED_Fill_In_Eval_Data_Path
metadata_path: REQUIRED_Fill_In_Metadata_Path
model_dir: REQUIRED_Fill_In_Model_Path
output_dir: REQUIRED_Fill_In_Output_Path
output_prediction_name: Fill_In_prediction_name_from_graph
"""
  IPython.get_ipython().set_next_input(content, replace=True)


def _output_evaluate_code_template(command, is_cloud, preprocessed_eval_data_path,
                                   metadata_path, model_dir, output_dir,
                                   output_prediction_name=None):
  # output_prediction_name is only useful for generating results analysis code.
  # It is only used in classification but not regression.
  content_header = _header_code(command)

  content_imports = \
"""import apache_beam as beam
from apache_beam.io import fileio
import google.cloud.ml as ml
import google.cloud.ml.analysis as analysis
import google.cloud.ml.dataflow.io.tfrecordio as tfrecordio
import google.cloud.ml.io as io
import json
import os
"""

  target_name, scenario = datalab.ml.Metadata(metadata_path).get_target_name_and_scenario()
  target_type = 'float_list' if scenario == 'continuous' else 'int64_list'
  content_definitions = \
"""def extract_values((example, prediction)):
  import tensorflow as tf
  tf_example = tf.train.Example()
  tf_example.ParseFromString(example.values()[0])
  feature_map = tf_example.features.feature
  values = {'target': feature_map['%s'].%s.value[0]}
  values.update(prediction)
  return values

OUTPUT_DIR = '%s'
%s
""" % (target_name, target_type, output_dir, _pipeline_definition_code(is_cloud, 'evaluate'))

  content_evaluation = \
"""eval_parameters = tfrecordio.TFRecordParameters(
    file_path_prefix='%s',
    file_name_suffix='',
    shard_file=False,
    compress_file=True)
eval_features = pipeline | io.LoadFeatures('LoadEvalFeatures', eval_parameters)
trained_model = pipeline | io.LoadModel('LoadModel', '%s')
evaluations = (eval_features | ml.Evaluate(trained_model, label='Evaluate')
    | beam.Map('ExtractEvaluationResults', extract_values))
eval_data_sink = beam.io.TextFileSink(os.path.join(OUTPUT_DIR, 'eval'), shard_name_template='')
evaluations | beam.Write('WriteEval', eval_data_sink)
""" % (preprocessed_eval_data_path, model_dir)

  output_analysis = (output_prediction_name is not None and scenario != 'continuous')
  content_analysis = ''
  if output_analysis:
    # TODO: 'score' is no longer needed when we use latest SDK.
    content_analysis = \
"""def make_data_for_analysis(values):
  return {
      'target': values['target'],
      'predicted': values['%s'],
      'score': 0.0,
  }

metadata = pipeline | io.LoadMetadata('%s')
analysis_source = evaluations | beam.Map('CreateAnalysisSource', make_data_for_analysis)
confusion_matrix, precision_recall, logloss = (analysis_source |
    analysis.AnalyzeModel('Analyze Model', metadata))
confusion_matrix_file = os.path.join(OUTPUT_DIR, 'analyze_cm.json')
confusion_matrix_sink = beam.io.TextFileSink(confusion_matrix_file, shard_name_template='')
confusion_matrix | beam.io.Write('WriteConfusionMatrix', confusion_matrix_sink)
""" % (output_prediction_name, metadata_path)

  content_run = """pipeline.run()"""

  content = \
"""
# header
%s
# imports
%s
# defines
%s
# evaluation
%s
# analysis
%s
# run pipeline
%s
""" % (content_header, content_imports, content_definitions,
       content_evaluation, content_analysis, content_run)

  if output_analysis:
    content += """
# View Confusion Matrix with the following code:
#
# import datalab.ml
# import yaml
# with ml.util._file.open_local_or_gcs(confusion_matrix_file, 'r') as f:
#   data = [yaml.load(line) for line in f.read().rstrip().split('\\n')]
# datalab.ml.ConfusionMatrix([d['predicted'] for d in data],
#                            [d['target'] for d in data],
#                            [d['count'] for d in data]).plot()
"""
  IPython.get_ipython().set_next_input(content, replace=True)


def _evaluate(args, cell):
  if not cell:
    _output_evaluate_template(args['cloud'])
    return

  env = datalab.utils.commands.notebook_environment()
  config = datalab.utils.commands.parse_config(cell, env)
  datalab.utils.commands.validate_config(config,
     ['preprocessed_eval_data_path', 'metadata_path', 'model_dir', 'output_dir'],
     optional_keys=['output_prediction_name'])
  command = '%%ml evaluate'
  if args['cloud']:
    command += ' --cloud'
  command += '\n' + cell
  _output_evaluate_code_template(command, args['cloud'], config['preprocessed_eval_data_path'],
      config['metadata_path'], config['model_dir'], config['output_dir'],
      output_prediction_name=config.get('output_prediction_name', None))


def _output_dataset_template(name):
  content = """%%ml dataset --name %s
source:
  data1: data_local_or_gcs_path
  data2: data_local_or_gcs_path
featureset: your-featureset-class-name
""" % name
  IPython.get_ipython().set_next_input(content, replace=True)


def _dataset(args, cell):
  if not cell:
    _output_dataset_template(args['name'])
    return
  env = datalab.utils.commands.notebook_environment()
  config = datalab.utils.commands.parse_config(cell, env)
  datalab.utils.commands.validate_config(config, ['source', 'featureset'],
      optional_keys=['format'])
  if config['featureset'] not in env:
    raise Exception('"%s" is not defined.' % config['featureset'])
  featureset_class = env[config['featureset']]
  format = config.get('format', 'csv')
  ds = datalab.ml.DataSet(featureset_class(), config['source'], format=format)
  env[args['name']] = ds


def _module(args, cell):
  if not cell:
    raise Exception('Expect code in cell.')
    return

  env = datalab.utils.commands.notebook_environment()
  if '_ml_modules_' not in env:
    modules = {}
    env['_ml_modules_'] = modules
  modules = env['_ml_modules_']
  modules[args['name']] = cell
  if args['main']:
    env['_ml_modules_main_'] = args['name']


def _package(args, cell):
  env = datalab.utils.commands.notebook_environment()
  if '_ml_modules_' not in env:
    raise Exception('No ml modules defined. Expect modules defined with "%%ml module"')
  package_path = datalab.ml.Packager().package(env['_ml_modules_'], args['name'])
  google.cloud.ml.util._file.create_directory(args['output'])
  dest = os.path.join(args['output'], os.path.basename(package_path))
  google.cloud.ml.util._file.copy_file(package_path, dest)
  os.remove(package_path)
  print 'Package created at %s.' % dest
