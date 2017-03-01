# Copyright 2017 Google Inc. All rights reserved.
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

# A copy of this file must be made in datalab_structured_data/setup.py

import datetime
import os
import re
from setuptools import setup



# The version is saved in an __init__ file.
def get_version():
    VERSIONFILE = os.path.join('datalab_structured_data/',
                               '__init__.py')
    if not os.path.isfile(VERSIONFILE):
      raise ValueError('setup.py: File not found %s' % VERSIONFILE)
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version string in %s.' % (VERSIONFILE,))


setup(
  name='datalab_structured_data',
  version=get_version(),
  packages=[
    'datalab_structured_data',
    'datalab_structured_data.trainer',
    'datalab_structured_data.preprocess',
    'datalab_structured_data.predict',
  ],
  description='Google Cloud Datalab Structured Data Package',
  author='Google',
  author_email='google-cloud-datalab-feedback@googlegroups.com',
  keywords=[
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
  long_description="""
  """,
  install_requires=[
    'tensorflow==1.0',
    'protobuf==3.1.0',
    'google-cloud-dataflow==0.5.5',
    'psutil>=4.3.0'  
  ],
  package_data={
  },
  data_files=[],
)
