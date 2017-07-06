#! /bin/bash
set -e

echo '*** Running code_free_ml test_analyze.py ***'
python test_analyze.py

echo '*** Running code_free_ml test_feature_transforms.py ***'
python test_feature_transforms.py

echo '*** Running code_free_ml test_transform.py ***'
python test_transform.py

echo '*** Running code_free_ml test_training.py ***'
python test_training.py

echo 'Finished code_free_ml run_all.sh!'