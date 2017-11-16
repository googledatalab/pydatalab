#!/bin/bash
AIRFLOW_CRON=temp_crontab.txt

export AIRFLOW_HOME=airflow
LOCAL_DAG_PATH=$AIRFLOW_HOME/dags

# TODO(rajivpb): These parameters should be gettable from the environment, i.e. they should have
# been set by the VM's create script. So eventually, remove these lines.
GCS_DAG_BUCKET="rajivpb-airflow-testing"
GCS_DAG_PATH="/dags"

apt-get --assume-yes install python-pip

pip install airflow
.local/bin/airflow initdb

# TODO(rajivpb): Replace this with pip install datalab
DATALAB_TAR=datalab-1.1.0.tar
gsutil cp gs://rajivpb-airflow-testing/$DATALAB_TAR $DATALAB_TAR
pip install $DATALAB_TAR
rm $DATALAB_TAR

# We append a gsutil rsync command to the cron file and have this run every minute to sync dags.
mkdir -p $LOCAL_DAG_PATH
crontab -l > $AIRFLOW_CRON
echo "* * * * * /usr/bin/gsutil rsync gs://$GCS_DAG_BUCKET/$GCS_DAG_PATH $LOCAL_DAG_PATH" >> $AIRFLOW_CRON
crontab $AIRFLOW_CRON
rm $AIRFLOW_CRON
