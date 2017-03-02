#! /bin/bash

rm -fr dist 
cp setup.py  mltoolbox/_structured_data/master_setup.py
python setup.py sdist


