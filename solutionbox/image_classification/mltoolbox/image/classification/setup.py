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

# To publish to PyPi use: python setup.py bdist_wheel upload -r pypi

import datetime
from setuptools import setup

minor = datetime.datetime.now().strftime("%y%m%d%H%M")
version = '0.1'

setup(
  name='mltoolbox_datalab_image_classification',
  namespace_packages=['mltoolbox'],
  version=version,
  packages=[
    'mltoolbox',
    'mltoolbox.image',
    'mltoolbox.image.classification',
  ],

  description='Google Cloud Datalab Inception Package',
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
    'pillow==3.4.1',
  ],
  package_data={
  }
)
