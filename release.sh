#!/bin/sh

# Build a distribution package
tsc --module amd --noImplicitAny --outdir datalab/notebook/static datalab/notebook/static/*.ts
python setup.py bdist_wheel
rm datalab/notebook/static/*.js


