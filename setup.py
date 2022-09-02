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

version = '1.2.1'

setup(
  name='datalab',
  version=version,
  namespace_packages=['google', 'datalab'],
  packages=[
    'google.datalab',
    'google.datalab.bigquery',
    'google.datalab.bigquery.commands',
    'google.datalab.commands',
    'google.datalab.contrib',
    'google.datalab.contrib.bigquery',
    'google.datalab.contrib.bigquery.commands',
    'google.datalab.contrib.bigquery.operators',
    'google.datalab.contrib.mlworkbench',
    'google.datalab.contrib.mlworkbench.commands',
    'google.datalab.contrib.pipeline',
    'google.datalab.contrib.pipeline.airflow',
    'google.datalab.contrib.pipeline.composer',
    'google.datalab.contrib.pipeline.commands',
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
      "Programming Language :: Python :: 3",
      "Development Status :: 7 - Inactive",
      "Environment :: Other Environment",
      "Intended Audience :: Developers",
      "License :: OSI Approved :: Apache Software License",
      "Operating System :: OS Independent",
      "Topic :: Software Development :: Libraries :: Python Modules"
  ],
  long_description_content_type='text/markdown',
  long_description="""\
Datalab is deprecated. [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench)
provides a notebook-based environment that offers capabilities beyond Datalab.
We recommend that you use Vertex AI Workbench for new projects and
[migrate your Datalab notebooks to Vertex AI Workbench](https://cloud.google.com/datalab/docs/resources/troubleshooting#migrate).
For more information, see
[Deprecation information](https://cloud.google.com/datalab/docs/resources/deprecation).
To get help migrating Datalab projects to Vertex AI Workbench see
[Get help](https://cloud.google.com/datalab/docs/resources/support#get-help).
  """,
  install_requires=[
    'configparser>=3.5.0',
    'mock>=2.0.0',
    'future>=0.16.0',
    'google-cloud-monitoring==0.31.1',
    'google-api-core>=1.10.0',
    'google-api-python-client>=1.6.2',
    'seaborn>=0.7.0',
    'plotly>=1.12.5',
    'httplib2>=0.10.3',
    'oauth2client>=2.2.0',
    'pandas>=0.22.0',
    'google_auth_httplib2>=0.0.2',
    'pandas-profiling==1.4.0',
    'python-dateutil>=2.5.0',
    'pytz>=2015.4',
    'pyyaml>=3.11',
    'requests>=2.9.1',
    'scikit-image>=0.13.0',
    'scikit-learn>=0.18.2',
    'ipykernel>=4.5.2',
    'psutil>=4.3.0',
    'jsonschema>=2.6.0',
    'six>=1.10.0',
    'urllib3>=1.22',
  ],
  extras_require={
    ':python_version == "2.7"': [
      'futures>=3.0.5',
    ]
  },
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
