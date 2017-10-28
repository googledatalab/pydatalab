#! /bin/bash
set -e

echo '*** Running code_free_ml test_analyze.py ***'
python test_analyze.py --verbose

echo '*** Running code_free_ml test_feature_transforms.py ***'
python test_feature_transforms.py --verbose

echo '*** Running code_free_ml test_transform.py ***'
python test_transform.py --verbose

echo '*** Running code_free_ml test_training.py ***'
python test_training.py --verbose

echo 'Finished code_free_ml run_all.sh!'