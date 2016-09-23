import google.cloud.ml.features as features
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pandas_profiling
from plotly.graph_objs import Bar, Figure, Histogram, Layout, Scatter, Scatter3d
from plotly.offline import iplot
from plotly import tools
import seaborn as sns
import tempfile

import datalab.utils

try:
  import IPython.core.display
except ImportError:
  raise Exception('This module can only be loaded in ipython.')


class DataSet(object):
  """Represents a dataset that can be explored through its 'analyze()' function.
     The data need to be able to fit in memory.
  """

  def __init__(self, feature_set, data_paths, format='csv'):
    """Initializes an instance of DataSet.

    Args:
      feature_set: A feature_set describing the data. The feature_set provides data types
          (for example, csv), column names, schema, data transformers, etc.
          This is the same class used in CloudML preprocessing.
      data_paths: A dictionary with {name: path} pair. All data need to have the same schema.
      format: the format of the data, currently only 'csv' or 'tsv'.
    """
    self._feature_set = feature_set
    if not isinstance(data_paths, dict):
      raise Exception('Expect "data_paths" to be a dictionary.')
    self._data_paths = data_paths
    if format == 'csv':
      self._delimiter = ','
    elif format=='tsv':
      self._delimiter = '\t'
    else:
      raise Exception('Unsupported format "%s"' % format)
    self._dataframes = {}
    self._raw_dataframes = {}
    self._concatenated_data_frame = None
    self._concatenated_raw_data_frame = None
    self._target_name = None
    self._key_name = None

  def _get_dataframe_type(self, column):
    if isinstance(column, features.NumericFeatureColumn):
      return np.float64
    if isinstance(column, features.TargetFeatureColumn) and column.is_numeric:
      return np.float64
    return str

  def _is_categorical_column(self, column):
    if isinstance(column, features.CategoricalFeatureColumn):
      return True
    if isinstance(column, features.TargetFeatureColumn) and not column.is_numeric:
      return True
    return False

  def _transform_data(self, df):
    df = df.copy(deep=True)
    for name, value in type(self._feature_set).__dict__.iteritems():
      for column in (value if (type(value) == list or type(value) == tuple) else [value]):
        if self._is_categorical_column(column):
          concatenated_column = self._concatenated_raw_data_frame[column.name]
          all_categories = concatenated_column.astype('category').cat.categories
          df[column.name] = pd.Categorical(df[column.name], categories=all_categories)
        if isinstance(column, features.NumericFeatureColumn):
          concatenated_column = self._concatenated_raw_data_frame[column.name]
          # Simulate metadata so we can create a transformer from CloudML features registry.
          transform_info = {
            'type': 'numeric',
            'transform': column.transform,
          }
          transform_info[column.transform] = column.transform_args
          transform_info['max'] = max(concatenated_column)
          transform_info['min'] = min(concatenated_column)
          transtormer = features._registries.transformation_registry \
              .get_transformer(transform_info)
          if column.transform == 'discretize':
            # Transformed data contains a one_of_k list so need to convert it back to index.
            # Categories needs to be num_of_buckets+2 to match the transformer behavior,
            # where it creates a smaller-than-min and a greater-than-max buckets.
            transformed = [transtormer.transform(x).index(1.0) for x in df[column.name]]
            df[column.name] = pd.Series(pd.Categorical(transformed,
                                                       categories=range(transtormer._buckets+2)))
          else:
            # TODO(qimingj): It is supposed to work with most transformers but still need to
            #                test them if new transformers become available.
            transformed = [transtormer.transform(x)[0] for x in df[column.name]]
            df[column.name] = transformed
    return df

  def _load_to_dataframes(self):
    if self._concatenated_raw_data_frame is not None:
      return  # Already loaded.

    # Step 1: Get schema from feature_set class.
    schema = {}
    for name, value in type(self._feature_set).__dict__.iteritems():
      for column in (value if (type(value) == list or type(value) == tuple) else [value]):
        if issubclass(type(column), features.FeatureColumn):
          if isinstance(column, features.TargetFeatureColumn):
            self._target_name = column.name
          if isinstance(column, features.KeyFeatureColumn):
            self._key_name = column.name
          data_type = self._get_dataframe_type(column)
          schema[column.name] = data_type
    if self._target_name is None:
      raise Exception('No target column found from feature_set')

    # Step 2: Load all non-text data into raw dataframes.
    for name, data_path in self._data_paths.iteritems():
      local_file = data_path
      if data_path.startswith('gs://'):
        local_file = tempfile.mktemp()
        datalab.utils.gcs_copy_file(data_path, local_file)
      self._raw_dataframes[name] = pd.read_csv(local_file,
                                               names=type(self._feature_set).csv_columns,
                                               dtype=schema,
                                               delimiter=self._delimiter)
      if data_path.startswith('gs://'):
        os.remove(local_file)
    self._concatenated_raw_data_frame = pd.concat(self._raw_dataframes.values())

    # Step 3: Transform the data.
    for name, raw_df in self._raw_dataframes.iteritems():
      self._dataframes[name] = self._transform_data(raw_df)
    self._concatenated_data_frame = pd.concat(self._dataframes.values())

  def _get_numeric_values(self, df, column_name):
    if str(df[column_name].dtype) == 'category':
      return df[column_name].cat.codes
    else:
      return df[column_name].values

  def _create_dummy_trace(self, x, y):
    # Dummy trace is needed for scatter plot to a) set the same x and y ranges across multiple
    # subplots, b) the categorical labels are sorted in the same way across multiple subplots
    # (the order of the categories depend on the order they appear in the data).
    # For a given axis, if it is categorical data, we draw one point for each category.
    # If it is numeric data, we draw min and max. Usually on x and y axises we don't have same
    # number of points, so we will pad one axis data.
    # Note: This needs to go away if plotly python supports setting ranges and specifying
    # category order across subplots.
    if str(self._concatenated_data_frame[x].dtype) == 'category':
      categories = self._concatenated_data_frame[x].cat.categories
      x_dummy = list(categories)
    else:
      x_dummy = [min(self._concatenated_data_frame[x]), max(self._concatenated_data_frame[x])]
    if str(self._concatenated_data_frame[y].dtype) == 'category':
      categories = self._concatenated_data_frame[y].cat.categories
      y_dummy = list(categories)
    else:
      y_dummy = [min(self._concatenated_data_frame[y]), max(self._concatenated_data_frame[y])]
    if len(x_dummy) > len(y_dummy):
      y_dummy = y_dummy + [y_dummy[-1]]*(len(x_dummy)-len(y_dummy))
    if len(x_dummy) < len(y_dummy):
      x_dummy = x_dummy + [x_dummy[-1]]*(len(y_dummy)-len(x_dummy))

    scatter_dummy = Scatter(
      x=x_dummy,
      y=y_dummy,
      showlegend=False,
      opacity=0, # Make it invisible.
      hoverinfo='none',
    )
    return scatter_dummy

  def _histogram(self, names, x):
    concatenated_numeric_values = self._get_numeric_values(self._concatenated_data_frame, x)
    start = min(concatenated_numeric_values)
    end = max(concatenated_numeric_values)
    size = 1 if str(self._concatenated_data_frame[x].dtype) == 'category' \
        else (max(concatenated_numeric_values) - min(concatenated_numeric_values)) / 10.0
    fig = tools.make_subplots(rows=1, cols=len(names), print_grid=False)
    histogram_index = 1
    for name in names:
      df = self._dataframes[name]
      numeric_values = self._get_numeric_values(df, x)
      text = df[x].cat.categories if str(df[x].dtype) == 'category' else None
      histogram = Histogram(
        name=name,
        x=numeric_values,
        xbins=dict(
          start=start,
          end=end,
          size=size,
        ),
        text=text,
      )
      fig.append_trace(histogram, 1, histogram_index)
      fig.layout['xaxis' + str(histogram_index)].title = x
      fig.layout['xaxis' + str(histogram_index)].range = [start, end]
      fig.layout['yaxis' + str(histogram_index)].title = 'count'
      histogram_index += 1
    fig.layout.width = min(500 * len(names), 1200)
    fig.layout.height = 500
    iplot(fig)

  def _scatter_plot(self, names, x, y, color):
    showscale = True if str(self._concatenated_data_frame[color].dtype) != 'category' else False
    cmin = min(self._get_numeric_values(self._concatenated_data_frame, color))
    cmax = max(self._get_numeric_values(self._concatenated_data_frame, color))
    fig = tools.make_subplots(rows=1, cols=len(names), print_grid=False)
    scatter_index = 1
    scatter_dummy = self._create_dummy_trace(x, y)
    for name in names:
      df = self._dataframes[name]
      text = ["x=%s y=%s target=%s" % (str(a),str(b),str(t)) for a,b,t
              in zip(df[x], df[y], df[color])]
      scatter = Scatter(
        name=name,
        x=df[x],
        y=df[y],
        mode='markers',
        text=text,
        hoverinfo='text',
        marker=dict(
          color=self._get_numeric_values(df, color),
          colorscale='Viridis',
          showscale=showscale,
          cmin=cmin,
          cmax=cmax,
        )
      )
      # Add dummy trace to set same ranges and categorical orders on subplots.
      fig.append_trace(scatter_dummy, 1, scatter_index)
      fig.append_trace(scatter, 1, scatter_index)
      fig.layout['xaxis' + str(scatter_index)].title = x
      fig.layout['yaxis' + str(scatter_index)].title = y
      scatter_index += 1
    fig.layout.width = min(500 * len(names), 1200)
    fig.layout.height = 500
    iplot(fig)

  def _scatter3d_plot(self, names, x, y, z, color):
    showscale = True if str(self._concatenated_data_frame[color].dtype) != 'category' else False
    cmin = min(self._get_numeric_values(self._concatenated_data_frame, color))
    cmax = max(self._get_numeric_values(self._concatenated_data_frame, color))
    specs = [[{'is_3d':True}]*len(self._dataframes)]
    fig = tools.make_subplots(rows=1, cols=len(names), specs=specs, print_grid=False)
    scatter3d_index = 1
    for name in names:
      df = self._dataframes[name]
      text = ["x=%s y=%s z=%s, target=%s" % (str(a),str(b),str(c),str(t)) for a,b,c,t
              in zip(df[x], df[y], df[z], df[color])]
      scatter3d = Scatter3d(
        name=name,
        x=df[x],
        y=df[y],
        z=df[z],
        mode='markers',
        text=text,
        hoverinfo='text',
        marker=dict(
          color=self._get_numeric_values(df, color),
          colorscale='Viridis',
          showscale=showscale,
          cmin=cmin,
          cmax=cmax,
        )
      )
      fig.append_trace(scatter3d, 1, scatter3d_index)
      fig.layout['scene' + str(scatter3d_index)].xaxis.title = x
      fig.layout['scene' + str(scatter3d_index)].yaxis.title = y
      fig.layout['scene' + str(scatter3d_index)].zaxis.title = z
      scatter3d_index += 1
    fig.layout.width = min(500 * len(names), 1200)
    fig.layout.height = 500
    iplot(fig)

  def _plot_x(self, names, x):
    self._histogram(names, x);
    if x != self._target_name:
      self._scatter_plot(names, x, self._target_name, self._target_name)

  def _plot_xy(self, names, x, y):
    self._scatter_plot(names, x, y, self._target_name)

  def _plot_xyz(self, names, x, y, z):
    self._scatter3d_plot(names, x, y, z, self._target_name)

  def profile(self, names=None, columns=None):
    """Print profiles of the dataset.

    Args:
      names: the names of the data to plot. Such as ['train']. If None, all data in the datasets
          will be used.
      columns: The list of column names to plot correlations. If None, all numeric columns
          will be used.
    """
    self._load_to_dataframes()
    if names is None:
      names = self._raw_dataframes.keys()
    html = ''
    for name in names:
      df = self._raw_dataframes[name]
      html += '<br/><br/><p style="text-align:center"><font size="6">' + \
              '<b>%s</b></font></p><br/><br/>' % name
      if columns is not None:
        df = df[columns]
      html += pandas_profiling.ProfileReport(df).html.replace('bootstrap', 'nonexistent')
    IPython.core.display.display_html(IPython.core.display.HTML(html))

  def analyze(self, names=None, columns=None):
    """Analyze the data and report results in IPython output cell. The results are based
           on preprocessed data as described in feature_set.
      columns: The list of column names to plot correlations. If None, all numeric columns
          will be used.
    Args:
      names: the names of the data to plot. Such as ['train']. If None, all data in the datasets
          will be used.
      columns: The list of names of columns to analyze.
          If one column provided, displays a scatter plot between the column and target
          column, and a histogram of the column.
          If two columns provided, displays a scatter plot between them,
          colored by target column.
          If threecolumns provided, displays a 3d scatter plot between them,
          colored by target column.
    Raises:
      Exception if any column names are not found in the data or the columns are text.
      Exception if columns are greater than 3 or less than 1.
    """
    self._load_to_dataframes()
    if columns is None:
      columns = [x for x in self._concatenated_data_frame 
          if str(self._concatenated_data_frame[x].dtype) != 'object']
    if len(columns) > 3 or len(columns) < 1:
      raise 'Found %d columns. ' % len(columns) + \
          'Use "columns" parameter to specify one, two or three columns.'
    for column_name in columns:
      if column_name not in self._concatenated_data_frame:
        raise Exception('Cannot find column "%s"' % column_name)
      if str(self._concatenated_data_frame[column_name].dtype) == 'object':
        raise Exception('Cannot analyze text column "%s"' % column_name)

    if names is None:
      names = self._dataframes.keys()
    if len(columns) == 1:
      self._plot_x(names, columns[0])
    elif len(columns) == 2:
      self._plot_xy(names, columns[0], columns[1])
    elif len(columns) == 3:
      self._plot_xyz(names, columns[0], columns[1], columns[2])

  def to_dataframes(self):
    """Get the transformed data as a DataFrames

    Returns: the transformed data in {name: dataframe} dictionary.
    """
    self._load_to_dataframes()
    return self._dataframes

  def plot(self, names=None, columns=None):
    """Plot specified columns correlation graphs, in n*n grids.

    Args:
      names: the names of the data to plot. Such as ['train']. If None, all data in the datasets
          will be used.
      columns: The list of column names to plot correlations. If None, all numeric columns
          will be used.
    """
    self.to_dataframes()
    if columns is not None and self._target_name not in columns:
      columns.append(self._target_name)
    if names is None:
      names = self._dataframes.keys()

    for name in names:
      df_correlation = self._dataframes[name].copy(deep=True)
      if self._key_name is not None:
        del df_correlation[self._key_name]
      if columns is not None:
        df_correlation = df_correlation[columns]
      for k in df_correlation.columns:
        if k == self._target_name:
          continue
        elif str(df_correlation[k].dtype) == 'object' or str(df_correlation[k].dtype) == 'category':
          # pairplot only works with numeric columns
          del df_correlation[k]
        else:
          # pairplot does not deal with missing values well. For now fillna(0).
          df_correlation[k] = df_correlation[k].fillna(0)
      # pairplot doesn't like categories with all numbers
      df_correlation[self._target_name] = map(lambda x: 'target ' + str(x), df_correlation[self._target_name])
      sns.set(style="ticks", color_codes=True)
      if str(self._concatenated_data_frame[self._target_name].dtype) == 'category':
        sns.pairplot(df_correlation, hue=self._target_name, dropna=True)
      else:
        sns.pairplot(df_correlation, dropna=True)
      plt.suptitle(name)
      plt.show()

