#! /bin/bash
set -e

python test_analyze.py
python test_feature_transforms.py
python test_transform.py
python test_training.py

echo 'No problems'