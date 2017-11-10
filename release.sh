#!/bin/bash -e
# Copyright 2017 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Compiles the typescript sources to javascript and submits the files
# to the pypi server specified as first parameter, defaults to testpypi
# In order to run this script locally, make sure you have the following:
# - A Python 3 environment (due to urllib issues)
# - Typescript installed
# - A configured ~/.pypirc containing your pypi/testpypi credentials with
#   the server names matching the name you're passing in. Do not include
#   the repository URLs in the config file, this has been deprecated.
# - Make sure the package version string in the setup.py file is updated.
#   It will get rejected by the server if it already exists
# - If this is a new release, make sure the release notes are updated
#   and create a new release tag

tsc --module amd --noImplicitAny datalab/notebook/static/*.ts
tsc --module amd --noImplicitAny google/datalab/notebook/static/*.ts

# Provide https://upload.pypi.org/legacy/ for prod binaries
server="${1:-https://test.pypi.python.org/pypi}"
echo "Submitting package to ${server}"

# Build and upload a distribution package
python setup.py sdist
twine upload --repository-url "${server}" dist/*

# Clean up
rm -f datalab/notebook/static/*.js
rm -f google/datalab/notebook/static/*.js
