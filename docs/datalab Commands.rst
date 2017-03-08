datalab Commands
=======================

.. attribute:: %monitoring
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

.. attribute:: %extension
.. parsed-literal::

  usage: %extension [-h] {mathjax} ...
  
  Load an extension into Datalab. Currently only mathjax is supported.
  
  positional arguments:
    {mathjax}   commands
      mathjax   Enabled MathJaX support in Datalab.
  
  optional arguments:
    -h, --help  show this help message and exit
  None

.. attribute:: %pymodule
.. parsed-literal::

  usage: pymodule [-h] [-n NAME]
  
  optional arguments:
    -h, --help            show this help message and exit
    -n NAME, --name NAME  the name of the python module to create and import
  None

.. attribute:: %projects
.. parsed-literal::

  usage: projects [-h] {list,set} ...
  
  positional arguments:
    {list,set}  commands
      list      List available projects.
      set       Set the default project.
  
  optional arguments:
    -h, --help  show this help message and exit
  None

.. attribute:: %sql
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

.. attribute:: %bigquery
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

.. attribute:: %storage
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

