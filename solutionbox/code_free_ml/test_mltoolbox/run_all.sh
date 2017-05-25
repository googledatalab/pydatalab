#! /bin/bash
set -e

python test_analyze_data.py
python test_transform_raw_data.py
python test_training.py

echo 'No problems'