google.datalab Commands
=======================

.. attribute:: %bq
.. parsed-literal::

  usage: %bq [-h]
             {datasets,tables,query,execute,extract,sample,dryrun,udf,datasource,load}
             ...
  
  Execute various BigQuery-related operations. Use "%bq <command> -h" for help
  on a specific command.
  
  positional arguments:
    {datasets,tables,query,execute,extract,sample,dryrun,udf,datasource,load}
                          commands
      datasets            Operations on BigQuery datasets
      tables              Operations on BigQuery tables
      query               Create or execute a BigQuery SQL query object,
                          optionally using other SQL objects, UDFs, or external
                          datasources. If a query name is not specified, the
                          query is executed.
      execute             Execute a BigQuery SQL query and optionally send the
                          results to a named table. The cell can optionally
                          contain arguments for expanding variables in the
                          query.
      extract             Extract a query or table into file (local or GCS)
      sample              Display a sample of the results of a BigQuery SQL
                          query. The cell can optionally contain arguments for
                          expanding variables in the query, if -q/--query was
                          used, or it can contain SQL for a query.
      dryrun              Execute a dry run of a BigQuery query and display
                          approximate usage statistics
      udf                 Create a named Javascript BigQuery UDF
      datasource          Create a named Javascript BigQuery external data
                          source
      load                Load data from GCS into a BigQuery table. If creating
                          a new table, a schema should be specified in YAML or
                          JSON in the cell body, otherwise the schema is
                          inferred from existing table.
  
  optional arguments:
    -h, --help            show this help message and exit
  None

.. attribute:: %chart
.. parsed-literal::

  usage: %chart [-h]
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

.. attribute:: %csv
.. parsed-literal::

  usage: csv [-h] {view} ...
  
  positional arguments:
    {view}      commands
      view      Browse CSV files without providing a schema. Each value is
                considered string type.
  
  optional arguments:
    -h, --help  show this help message and exit
  None

.. attribute:: %datalab
.. parsed-literal::

  usage: %datalab [-h] {config,project} ...
  
  Execute operations that apply to multiple Datalab APIs. Use "%datalab
  <command> -h" for help on a specific command.
  
  positional arguments:
    {config,project}  commands
      config          List or set API-specific configurations.
      project         Get or set the default project ID
  
  optional arguments:
    -h, --help        show this help message and exit
  None

.. attribute:: %gcs
.. parsed-literal::

  usage: %gcs [-h] {copy,create,delete,list,read,view,write} ...
  
  Execute various Google Cloud Storage related operations. Use "%gcs <command>
  -h" for help on a specific command.
  
  positional arguments:
    {copy,create,delete,list,read,view,write}
                          commands
      copy                Copy one or more Google Cloud Storage objects to a
                          different location.
      create              Create one or more Google Cloud Storage buckets.
      delete              Delete one or more Google Cloud Storage buckets or
                          objects.
      list                List buckets in a project, or contents of a bucket.
      read                Read the contents of a Google Cloud Storage object
                          into a Python variable.
      view                View the contents of a Google Cloud Storage object.
      write               Write the value of a Python variable to a Google Cloud
                          Storage object.
  
  optional arguments:
    -h, --help            show this help message and exit
  None

.. attribute:: %sd
.. parsed-literal::

  usage: %sd [-h] {monitoring} ...
  
  Execute various Stackdriver related operations. Use "%sd <stackdriver_product>
  -h" for help on a specific Stackdriver product.
  
  positional arguments:
    {monitoring}  commands
      monitoring  Execute Stackdriver monitoring related operations. Use "sd
                  monitoring <command> -h" for help on a specific command
  
  optional arguments:
    -h, --help    show this help message and exit
  None

