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

import datetime
import platform
from setuptools import setup
import pip

minor = datetime.datetime.now().strftime("%y%m%d%H%M")
version = '0.1.' + minor

setup(
  name='datalab',
  version=version,
  namespace_packages=['datalab'],
  packages=[
    'datalab.bigquery',
    'datalab.bigquery.commands',
    'datalab.context',
    'datalab.context.commands',
    'datalab.data',
    'datalab.data.commands',
    'datalab.kernel',
    'datalab.mlalpha',
    'datalab.mlalpha.commands',
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
  url='https://github.com/googledatalab/datalab',
  download_url='https://github.com/googledatalab/datalab/tarball/0.1',
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
    'future==0.15.2',
    'futures==3.0.5',
    'google-cloud==0.19.0',
    'google-api-python-client==1.5.1',
    'seaborn==0.7.0',
    'plotly==1.12.5',
    'httplib2==0.9.2',
    'oauth2client==2.2.0',
    'psutil==4.3.0',
    'pandas>=0.17.1',
    'pandas-profiling>=1.0.0a2',
    'python-dateutil==2.5.0',
    'pytz>=2015.4',
    'pyyaml==3.11',
    'requests==2.9.1',
    'ipykernel==4.4.1',
  ],
  package_data={
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
      ]
  }
)

tensorflow_path = None
if platform.system() == 'Darwin':
  tensorflow_path = 'https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0-py2-none-any.whl'
elif platform.system() == 'Linux':
  if platform.linux_distribution()[0] == 'Ubuntu':
    tensorflow_path = 'https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl'
  elif platform.linux_distribution()[0] == 'Debian':
    tensorflow_path = 'https://storage.googleapis.com/tensorflow/linux/cpu/debian/jessie/tensorflow-0.11.0-cp27-none-linux_x86_64.whl'

# install tensorflow
if not tensorflow_path:
  print("""Warning: could not find tensorflow build for your OS.
  Please go to https://www.tensorflow.org/get_started/os_setup to see install options""")
else:
  pip.main(['install', tensorflow_path])

# install cloud ml sdk
pip.main(['install', 'https://storage.googleapis.com/cloud-ml/sdk/cloudml.latest.tar.gz'])
