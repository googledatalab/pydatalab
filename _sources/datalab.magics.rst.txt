datalab.magics
=================

.. attribute:: extension
.. parsed-literal::

  usage: %extension [-h] {mathjax} ...
  
  Load an extension into Datalab. Currently only mathjax is supported.
  
  positional arguments:
    {mathjax}   commands
      mathjax   Enabled MathJaX support in Datalab.
  
  optional arguments:
    -h, --help  show this help message and exit
  None

.. attribute:: pymodule
.. parsed-literal::

  usage: pymodule [-h] [-n NAME]
  
  optional arguments:
    -h, --help            show this help message and exit
    -n NAME, --name NAME  the name of the python module to create and import
  None

.. attribute:: chart
.. parsed-literal::

  usage: %%chart [-h]
                 {annotation,area,bars,bubbles,calendar,candlestick,columns,combo,gauge,geo,heatmap,histogram,line,map,org,paged_table,pie,sankey,scatter,stepped_area,table,timeline,treemap}
                 ...
  
  Generate an inline chart using Google Charts using the data in a Table, Query,
  dataframe, or list. Numerous types of charts are supported. Options for the
  charts can be specified in the cell body using YAML or JSON.
  
  positional arguments:
    {annotation,area,bars,bubbles,calendar,candlestick,columns,combo,gauge,geo,heatmap,histogram,line,map,org,paged_table,pie,sankey,scatter,stepped_area,table,timeline,treemap}
                          commands
      annotation          Generate a annotation chart.
      area                Generate a area chart.
      bars                Generate a bars chart.
      bubbles             Generate a bubbles chart.
      calendar            Generate a calendar chart.
      candlestick         Generate a candlestick chart.
      columns             Generate a columns chart.
      combo               Generate a combo chart.
      gauge               Generate a gauge chart.
      geo                 Generate a geo chart.
      heatmap             Generate a heatmap chart.
      histogram           Generate a histogram chart.
      line                Generate a line chart.
      map                 Generate a map chart.
      org                 Generate a org chart.
      paged_table         Generate a paged_table chart.
      pie                 Generate a pie chart.
      sankey              Generate a sankey chart.
      scatter             Generate a scatter chart.
      stepped_area        Generate a stepped_area chart.
      table               Generate a table chart.
      timeline            Generate a timeline chart.
      treemap             Generate a treemap chart.
  
  optional arguments:
    -h, --help            show this help message and exit
  None

.. attribute:: csv
.. parsed-literal::

  usage: csv [-h] {view} ...
  
  positional arguments:
    {view}      commands
      view      Browse CSV files without providing a schema. Each value is
                considered string type.
  
  optional arguments:
    -h, --help  show this help message and exit
  None

.. attribute:: bigquery
.. parsed-literal::

  usage: bigquery [-h]
                  {sample,create,delete,dryrun,udf,execute,pipeline,table,schema,datasets,tables,extract,load}
                  ...
  
  Execute various BigQuery-related operations. Use "%bigquery <command> -h" for
  help on a specific command.
  
  positional arguments:
    {sample,create,delete,dryrun,udf,execute,pipeline,table,schema,datasets,tables,extract,load}
                          commands
      sample              Display a sample of the results of a BigQuery SQL
                          query. The cell can optionally contain arguments for
                          expanding variables in the query, if -q/--query was
                          used, or it can contain SQL for a query.
      create              Create a dataset or table.
      delete              Delete a dataset or table.
      dryrun              Execute a dry run of a BigQuery query and display
                          approximate usage statistics
      udf                 Create a named Javascript BigQuery UDF
      execute             Execute a BigQuery SQL query and optionally send the
                          results to a named table. The cell can optionally
                          contain arguments for expanding variables in the
                          query.
      pipeline            Define a deployable pipeline based on a BigQuery
                          query. The cell can optionally contain arguments for
                          expanding variables in the query.
      table               View a BigQuery table.
      schema              View a BigQuery table or view schema.
      datasets            List the datasets in a BigQuery project.
      tables              List the tables in a BigQuery project or dataset.
      extract             Extract BigQuery query results or table to GCS.
      load                Load data from GCS into a BigQuery table.
  
  optional arguments:
    -h, --help            show this help message and exit
  None

.. attribute:: mlalpha
.. parsed-literal::

  usage: mlalpha [-h]
                 {train,jobs,summary,features,predict,model,deploy,delete,preprocess,evaluate,dataset,module,package}
                 ...
  
  Execute various ml-related operations. Use "%%mlalpha <command> -h" for help
  on a specific command.
  
  positional arguments:
    {train,jobs,summary,features,predict,model,deploy,delete,preprocess,evaluate,dataset,module,package}
                          commands
      train               Run a training job.
      jobs                List jobs in a project.
      summary             List or view summary events.
      features            Generate featureset class template.
      predict             Get prediction results given data instances.
      model               List or view models.
      deploy              Deploy a model.
      delete              Delete a model or a model version.
      preprocess          Generate preprocess code template.
      evaluate            Generate evaluate code template.
      dataset             Define dataset to explore data.
      module              Define a trainer module.
      package             Create a trainer package from all modules defined with
                          %mlalpha module.
  
  optional arguments:
    -h, --help            show this help message and exit
  None

.. attribute:: tensorboard
.. parsed-literal::

  usage: tensorboard [-h] {list,start,stop} ...
  
  Execute tensorboard operations. Use "%tensorboard <command> -h" for help on a
  specific command.
  
  positional arguments:
    {list,start,stop}  commands
      list             List running TensorBoard instances.
      start            Start a TensorBoard server with the given logdir.
      stop             Stop a TensorBoard server with the given pid.
  
  optional arguments:
    -h, --help         show this help message and exit
  None

.. attribute:: storage
.. parsed-literal::

  usage: storage [-h] {copy,create,delete,list,read,view,write} ...
  
  Execute various storage-related operations. Use "%storage <command> -h" for
  help on a specific command.
  
  positional arguments:
    {copy,create,delete,list,read,view,write}
                          commands
      copy                Copy one or more GCS objects to a different location.
      create              Create one or more GCS buckets.
      delete              Delete one or more GCS buckets or objects.
      list                List buckets in a project, or contents of a bucket.
      read                Read the contents of a storage object into a Python
                          variable.
      view                View the contents of a storage object.
      write               Write the value of a Python variable to a storage
                          object.
  
  optional arguments:
    -h, --help            show this help message and exit
  None

.. attribute:: sql
.. parsed-literal::

  usage: %%sql [-h] [-m MODULE] [-d {legacy,standard}] [-b BILLING]
  
  Create a named SQL module with one or more queries.
  
  The cell body should contain an optional initial part defining the default
  values for the variables, if any, using Python code, followed by one or more
  queries.
  
  Queries should start with 'DEFINE QUERY <name>' in order to bind them to
  <module name>.<query name> in the notebook (as datalab.data.SqlStament instances).
  The final query can optionally omit 'DEFINE QUERY <name>', as using the module
  name in places where a SqlStatement is expected will resolve to the final query
  in the module.
  
  Queries can refer to variables with '$<name>', as well as refer to other queries
  within the same module, making it easy to compose nested queries and test their
  parts.
  
  The Python code defining the variable default values can assign scalar or list/tuple values to
  variables, or one of the special functions 'datestring' and 'source'.
  
  When a variable with a 'datestring' default is expanded it will expand to a formatted
  string based on the current date, while a 'source' default will expand to a table whose
  name is based on the current date.
  
  datestring() takes two named arguments, 'format' and 'offset'. The former is a
  format string that is the same as for Python's time.strftime function. The latter
  is a string containing a comma-separated list of expressions such as -1y, +2m,
  etc; these are offsets from the time of expansion that are applied in order. The
  suffix (y, m, d, h, M) correspond to units of years, months, days, hours and
  minutes, while the +n or -n prefix is the number of units to add or subtract from
  the time of expansion. Three special values 'now', 'today' and 'yesterday' are
  also supported; 'today' and 'yesterday' will be midnight UTC on the current date
  or previous days date.
  
  source() can take a 'name' argument for a fixed table name, or 'format' and 'offset'
  arguments similar to datestring(), but unlike datestring() will resolve to a Table
  with the specified name.
  
  optional arguments:
    -h, --help            show this help message and exit
    -m MODULE, --module MODULE
                          The name for this SQL module
    -d {legacy,standard}, --dialect {legacy,standard}
                          BigQuery SQL dialect
    -b BILLING, --billing BILLING
                          BigQuery billing tier

.. attribute:: monitoring
.. parsed-literal::

  usage: monitoring [-h] {list} ...
  
  Execute various Monitoring-related operations. Use "%monitoring <command> -h"
  for help on a specific command.
  
  positional arguments:
    {list}      commands
      list      List the metrics or resource types in a monitored project.
  
  optional arguments:
    -h, --help  show this help message and exit
  None

.. attribute:: projects
.. parsed-literal::

  usage: projects [-h] {list,set} ...
  
  positional arguments:
    {list,set}  commands
      list      List available projects.
      set       Set the default project.
  
  optional arguments:
    -h, --help  show this help message and exit
  None

