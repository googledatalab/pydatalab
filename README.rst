datalab
=======

Google Datalab Library

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

