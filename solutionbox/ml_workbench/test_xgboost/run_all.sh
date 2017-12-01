#! /bin/bash
set -e

echo '*** Running xgboost test_analyze.py ***'
python test_analyze.py --verbose

echo '*** Running xgboost test_transform.py ***'
python test_transform.py --verbose

echo 'Finished xgboost run_all.sh!'
