# datalab [![Build Status](https://travis-ci.org/googledatalab/pydatalab.svg?branch=master)](https://travis-ci.org/googledatalab/pydatalab) [![PyPI Package](https://img.shields.io/pypi/v/datalab.svg)](https://pypi.python.org/pypi/datalab)

[Google Cloud Datalab](https://cloud.google.com/datalab/) Python package. Used in
[Google Cloud Datalab](https://github.com/GoogleCloudPlatform/datalab) and can
be used in [Jupyter Notebook](http://jupyter.org/).

This adds a number of Python modules such as `google.datalab.bigquery`,
`google.datalab.storage`, etc, for accessing
[Google Cloud Platform services](https://cloud.google.com/) as well as adding
some new cell magics such as `%chart`, `%bigquery`, `%storage`, etc.

See
[https://github.com/googledatalab/notebooks](https://github.com/googledatalab/notebooks)
for samples of using this package.

## Installation

This package is available on PyPI as `datalab`:

    pip install datalab

## Using in Jupyter

After datalab installation, enable datalab's frontend in Jupyter by running:

    jupyter nbextension install --py datalab.notebook --sys-prefix

See further details [Jupyter Kernel and Notebook Extensions](https://github.com/googledatalab/pydatalab/wiki/Jupyter-Kernel-and-Notebook-Extensions).

Then in a notebook cell, enable datalab's magics with:

    %load_ext google.datalab.kernel
    
(Note: If you hit an error "module traceback cannot be imported", try setting the following environment variable: CLOUDSDK_PYTHON_SITEPACKAGES=1)

Alternatively add this to your `ipython_config.py` file in your profile:

    c = get_config()
    c.InteractiveShellApp.extensions = [
        'google.datalab.kernel'
    ]

You will typically put this under `~/.ipython/profile_default`. See
[the IPython docs](http://ipython.readthedocs.io/en/stable/development/config.html)
for more about IPython profiles.

If you want to access Google Cloud Platform services such as BigQuery, you
will also need to install [gcloud](https://cloud.google.com/sdk/gcloud). You
will need to use `gcloud` to authenticate; e.g. with:

    gcloud auth login

You will also need to set the project ID to use; either set a `PROJECT_ID`
environment variable to the project name, or call
`set_datalab_project_id(name)` from within your notebook.

## Documentation

You can read the Sphinx generated docs at:
[http://googledatalab.github.io/pydatalab/](http://googledatalab.github.io/pydatalab/)

## Development installation

If you'd like to work on the package, it's useful to be able to install from
source. You will need the
[Typescript compiler](https://www.typescriptlang.org/) installed.

First:

    git clone https://github.com/googledatalab/pydatalab.git
    cd pydatalab

Then do one of the folowing:

    ./install-virtualenv.sh  # For use in Python virtual environments
    ./install-no-virtualenv.sh  # For installing in a non-virtual environment

You can ignore the message about running `jupyter nbextension enable`; it is
not required.

