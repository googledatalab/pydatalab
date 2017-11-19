#!/bin/bash
PROJECT_ID=cloud-ml-dev
export AIRFLOW_HOME=/airflow
export AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=False
AIRFLOW__CORE__LOAD_EXAMPLES=False
GCS_DAG_BUCKET=$PROJECT_ID-datalab-airflow

apt-get --assume-yes install python-pip

mkdir $AIRFLOW_HOME
chmod a+rwx $AIRFLOW_HOME

# TODO(rajivpb): Replace this with 'pip install datalab'
DATALAB_TAR=datalab-1.1.0.tar
gsutil cp gs://datalab-pipelines/$DATALAB_TAR $DATALAB_TAR
pip install $DATALAB_TAR
rm $DATALAB_TAR

airflow initdb
airflow scheduler &

# We append a gsutil rsync command to the cron file and have this run every minute to sync dags.
AIRFLOW_CRON=temp_crontab.txt
crontab -l > $AIRFLOW_CRON
DAG_FOLDER="dags"
LOCAL_DAG_PATH=$AIRFLOW_HOME/$DAG_FOLDER
mkdir -p $LOCAL_DAG_PATH
chmod a+rwx $LOCAL_DAG_PATH
echo "* * * * * gsutil rsync gs://$GCS_DAG_BUCKET/$DAG_FOLDER $LOCAL_DAG_PATH" >> $AIRFLOW_CRON
crontab $AIRFLOW_CRON
rm $AIRFLOW_CRON

