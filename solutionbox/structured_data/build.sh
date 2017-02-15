#! /bin/bash


rm -fr dist 
cp setup.py  datalab_solutions/structured_data/
python setup.py sdist


