#!/bin/bash
export AIRFLOW_HOME=/etc/airflow
PROJECT_ID=cloud-ml-dev
GCS_DAG_BUCKET=$PROJECT_ID-datalab-airflow

GSUTIL=gsutil
GCLOUD=gcloud

apt-get --assume-yes install python-pip
pip install airflow
airflow initdb

# TODO(rajivpb): Replace this with 'pip install datalab'
DATALAB_TAR=datalab-1.1.0.tar
$GSUTIL cp gs://$GCS_DAG_BUCKET/$DATALAB_TAR $DATALAB_TAR
pip install $DATALAB_TAR
rm $DATALAB_TAR

# We append a gsutil rsync command to the cron file and have this run every minute to sync dags.
AIRFLOW_CRON=temp_crontab.txt
crontab -l > $AIRFLOW_CRON
DAG_FOLDER="dags"
LOCAL_DAG_PATH=$AIRFLOW_HOME/$DAG_FOLDER
mkdir -p $LOCAL_DAG_PATH

# TODO(rajivpb): Temporarily give other users permissions to peek into the dag folder for debugging
chmod a+rwx $AIRFLOW_HOME
chmod a+rwx $LOCAL_DAG_PATH

echo "* * * * * gsutil rsync gs://$GCS_DAG_BUCKET/$DAG_FOLDER $LOCAL_DAG_PATH" >> $AIRFLOW_CRON
crontab $AIRFLOW_CRON
rm $AIRFLOW_CRON
