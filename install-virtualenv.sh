#!/bin/sh

# Build a distribution package
tsc --module amd --noImplicitAny --outdir datalab/notebook/static datalab/notebook/static/*.ts
pip install .
jupyter nbextension install --py datalab.notebook --sys-prefix
rm datalab/notebook/static/*.js
