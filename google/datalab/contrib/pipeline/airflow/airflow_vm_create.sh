#!/usr/bin/env bash
PROJECT_ID=${1:-cloud-ml-dev}
VM_NAME=${2:-instance-16}
ZONE=us-central1-b

GSUTIL=gsutil
GCLOUD=gcloud

# Make bucket
GCS_DAG_BUCKET=$PROJECT_ID-datalab-airflow
$GSUTIL mb gs://$GCS_DAG_BUCKET

# Copy startup script to the new bucket
SRC_BUCKET=rajivpb-airflow-testing
AIRFLOW_VM_STARTUP_SCRIPT_SRC=gs://$SRC_BUCKET/airflow_vm_startup.sh
AIRFLOW_VM_STARTUP_SCRIPT=gs://$GCS_DAG_BUCKET/airflow_vm_startup.sh
$GSUTIL cp $AIRFLOW_VM_STARTUP_SCRIPT_SRC $AIRFLOW_VM_STARTUP_SCRIPT

# Copy datalab.tar to the new bucket
# TODO(rajivpb): This should go away after we release pydatalab and pip install it from PyPI
DATALAB_TAR=datalab-1.1.0.tar
$GSUTIL cp gs://$SRC_BUCKET/$DATALAB_TAR gs://$GCS_DAG_BUCKET/$DATALAB_TAR

# The default values here were grabbed from Pantheon's "command-line" button for creating a VM
$GCLOUD beta compute --project $PROJECT_ID instances create $VM_NAME \
    --zone $ZONE \
    --machine-type "n1-standard-1" \
    --network "default" \
    --maintenance-policy "MIGRATE" \
    --scopes "https://www.googleapis.com/auth/cloud-platform" \
    --min-cpu-platform "Automatic" \
    --image "debian-9-stretch-v20171025" \
    --image-project "debian-cloud" \
    --boot-disk-size "10" \
    --boot-disk-type "pd-standard" \
    --boot-disk-device-name $VM_NAME \
    --metadata startup-script-url=$AIRFLOW_VM_STARTUP_SCRIPT \

# TODO(rajivpb): To be deleted; left here only for convenience
sleep 30
$GCLOUD compute --project $PROJECT_ID ssh --zone $ZONE $VM_NAME
