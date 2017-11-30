#! /bin/bash
set -e

echo '*** Running tensorflow test_analyze.py ***'
python test_analyze.py --verbose

echo '*** Running tensorflow test_feature_transforms.py ***'
python test_feature_transforms.py --verbose

echo '*** Running tensorflow test_transform.py ***'
python test_transform.py --verbose

echo '*** Running tensorflow test_training.py ***'
python test_training.py --verbose

echo 'Finished tensorflow run_all.sh!'
