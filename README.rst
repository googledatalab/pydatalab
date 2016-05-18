datalab
=======

Google Datalab Library. Used in `Google Cloud Datalab
<https://github.com/GoogleCloudPlatform/datalab>`_ and can be used
in `Jupyter Notebook
<http://jupyter.org/>`_.

Installation
------------

    python setup.py install
    jupyter nbextension install --py datalab.notebook --sys-prefix

The `--sys-prefix` should be omitted if not in a virtualenv.

Using in Jupyter
----------------

In a notebook cell, enable with:

    %load_ext datalab.kernel

Alternatively add this to your `ipython_config.py` file in your profile:


    c.InteractiveShellApp.extensions = [
        'datalab.kernel'
    ]

See `http://ipython.readthedocs.io/en/stable/development/config.html
<http://ipython.readthedocs.io/en/stable/development/config.html>`_
for more about IPython profiles.

If you want to access Google Cloud Platform services such as BigQuery,
you will also need to install `gcloud
<https://cloud.google.com/sdk/gcloud/>`_. You will need to use gcloud
to authenticate; e.g. with:

    gcloud auth login

You will also need to set the project ID to use; either set a PROJECT_ID 
environment variable to the project name, or call set_project_id(name)
from within your notebook.

