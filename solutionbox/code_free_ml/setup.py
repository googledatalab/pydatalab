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

import os
import re
from setuptools import setup


setup(
  name='mltoolbox_code_free',
  namespace_packages=['mltoolbox'],
  version='1.0.0',
  packages=[
    'mltoolbox',
    'mltoolbox.code_free_ml'
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
  ],
  package_data={
    'mltoolbox.code_free_ml': ['data/*'],
  },
  data_files=[],
)
