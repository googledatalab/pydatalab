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

# To publish to PyPi use: python setup.py bdist_wheel upload -r pypi

from setuptools import setup

version = '1.1.0'

setup(
  name='datalab',
  version=version,
  namespace_packages=['google', 'datalab'],
  packages=[
    'google.datalab',
    'google.datalab.bigquery',
    'google.datalab.bigquery.commands',
    'google.datalab.commands',
    'google.datalab.data',
    'google.datalab.kernel',
    'google.datalab.ml',
    'google.datalab.notebook',
    'google.datalab.stackdriver',
    'google.datalab.stackdriver.commands',
    'google.datalab.stackdriver.monitoring',
    'google.datalab.storage',
    'google.datalab.storage.commands',
    'google.datalab.utils',
    'google.datalab.utils.commands',
    'google.datalab.utils.facets',
    'google.datalab.contrib',
    'google.datalab.contrib.mlworkbench',
    'google.datalab.contrib.mlworkbench.commands',
    'datalab.bigquery',
    'datalab.bigquery.commands',
    'datalab.context',
    'datalab.context.commands',
    'datalab.data',
    'datalab.data.commands',
    'datalab.kernel',
    'datalab.notebook',
    'datalab.stackdriver',
    'datalab.stackdriver.commands',
    'datalab.stackdriver.monitoring',
    'datalab.storage',
    'datalab.storage.commands',
    'datalab.utils',
    'datalab.utils.commands'
  ],
  description='Google Cloud Datalab',
  author='Google',
  author_email='google-cloud-datalab-feedback@googlegroups.com',
  url='https://github.com/googledatalab/pydatalab',
  download_url='https://github.com/googledatalab/pydatalab/archive/v1.1.0.zip',
  keywords=[
    'Google',
    'GCP',
    'GCS',
    'bigquery'
  ],
  license="Apache Software License",
  classifiers=[
      "Programming Language :: Python",
      "Programming Language :: Python :: 2",
      "Development Status :: 4 - Beta",
      "Environment :: Other Environment",
      "Intended Audience :: Developers",
      "License :: OSI Approved :: Apache Software License",
      "Operating System :: OS Independent",
      "Topic :: Software Development :: Libraries :: Python Modules"
  ],
  long_description="""\
Support package for Google Cloud Datalab. This provides cell magics and Python APIs
for accessing Google's Cloud Platform services such as Google BigQuery.
  """,
  install_requires=[
    'mock==2.0.0',
    'future==0.16.0',
    'futures==3.0.5',
    'google-cloud==0.19.0',
    'google-api-python-client==1.6.2',
    'seaborn==0.7.0',
    'plotly==1.12.5',
    'httplib2==0.10.3',
    'oauth2client==2.2.0',
    'pandas>=0.17.1',
    'pandas-profiling>=1.0.0a2',
    'python-dateutil==2.5.0',
    'pytz>=2015.4',
    'pyyaml==3.11',
    'requests==2.9.1',
    'scikit-learn==0.18.2',
    'ipykernel==4.5.2',
    'psutil==4.3.0',
    'jsonschema==2.6.0',
  ],
  package_data={
    'google.datalab.notebook': [
        'static/bigquery.css',
        'static/bigquery.js',
        'static/charting.css',
        'static/charting.js',
        'static/job.css',
        'static/job.js',
        'static/element.js',
        'static/style.js',
        'static/visualization.js',
        'static/codemirror/mode/bigquery.js',
        'static/parcoords.js',
        'static/extern/d3.parcoords.js',
        'static/extern/d3.parcoords.css',
        'static/extern/sylvester.js',
        'static/extern/lantern-browser.html',
        'static/extern/facets-jupyter.html',
      ],
    'datalab.notebook': [
        'static/bigquery.css',
        'static/bigquery.js',
        'static/charting.css',
        'static/charting.js',
        'static/job.css',
        'static/job.js',
        'static/element.js',
        'static/style.js',
        'static/visualization.js',
        'static/codemirror/mode/sql.js',
        'static/parcoords.js',
        'static/extern/d3.parcoords.js',
        'static/extern/d3.parcoords.css',
        'static/extern/sylvester.js',
        'static/extern/lantern-browser.html',
        'static/extern/facets-jupyter.html',
      ]
  }
)
