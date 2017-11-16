#!/bin/bash
LOCAL_DAG_PATH=airflow/dags
TEMP_CRON_FILE=temp_crontab.txt

# TODO(rajivpb): These parameters should be gettable from the environment, i.e. they should have
# been set by the VM's create script. So eventually, remove these lines.
GCS_DAG_BUCKET="rajivpb-airflow-testing"
GCS_DAG_PATH="/dags"

export AIRFLOW_HOME=airflow

apt-get --assume-yes install python-pip

pip install airflow
.local/bin/airflow initdb

# TODO(rajivpb): Replace this with pip install datalab
gsutil cp gs://rajivpb-airflow-testing/datalab-1.1.0.tar .
pip install datalab-1.1.0.tar
rm datalab-1.1.0.tar

# We append a gsutil rsync command to the cron file and have this run every minute to sync dags.
mkdir -p $LOCAL_DAG_PATH
crontab -l > $TEMP_CRON_FILE
echo "* * * * * /usr/bin/gsutil rsync gs://$GCS_DAG_BUCKET/$GCS_DAG_PATH $LOCAL_DAG_PATH" >> $TEMP_CRON_FILE
crontab $TEMP_CRON_FILE
rm $TEMP_CRON_FILE
