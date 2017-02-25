#! /bin/bash


rm -fr dist 
cp setup.py  datalab_solutions/structured_data/master_setup.py
python setup.py sdist


