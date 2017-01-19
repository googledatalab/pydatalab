# datalab

Google datalab Python package. Used in [Google Cloud Datalab]
(https://github.com/GoogleCloudPlatform/datalab) and can be used
in [Jupyter Notebook](http://jupyter.org/).

This adds a number of Python modules such as datalab.bigquery, 
datalab.storage, etc, for accessing [Google Cloud Platform services]
(https://cloud.google.com/) as well as adding some new cell magics such as `%chart`,
`%bigquery`, `%storage`, etc.

See [https://github.com/googledatalab/notebooks](https://github.com/googledatalab/notebooks) 
for samples of using this package.

## Prerequisites

You will need the [Typscript compiler](https://www.typescriptlang.org/) installed.
In future we should be installable from PyPI.

## Installation

First:

    git clone https://github.com/googledatalab/pydatalab.git
    cd pydatalab

Then do one of the folowing:

    ./install-virtualenv.sh  # For use in Python virtual environments
    ./install-no-virtualenv.sh  # For installing in a non-virtual environment

You can ignore the message about running `jupyter nbextension enable`; it is not required.

## Using in Jupyter

In a notebook cell, enable with:

    %load_ext datalab.kernel

Alternatively add this to your `ipython_config.py` file in your profile:

    c = get_config()
    c.InteractiveShellApp.extensions = [
        'datalab.kernel'
    ]

You will typically put this under `~/.ipython/profile_default`. 
See [http://ipython.readthedocs.io/en/stable/development/config.html]
(http://ipython.readthedocs.io/en/stable/development/config.html)
for more about IPython profiles.

If you want to access Google Cloud Platform services such as BigQuery,
you will also need to install [gcloud]
(https://cloud.google.com/sdk/gcloud). You will need to use gcloud
to authenticate; e.g. with:

    gcloud auth login

You will also need to set the project ID to use; either set a `PROJECT_ID`
environment variable to the project name, or call `set_datalab_project_id(name)`
from within your notebook.

## Documentation
You can read the Sphinx generated docs at: [http://googledatalab.github.io/pydatalab/](http://googledatalab.github.io/pydatalab/)
