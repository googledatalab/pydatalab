#!/usr/bin/env bash
PROJECT_ID=${1:-cloud-ml-dev}
VM_NAME=${2:-instance-16}
ZONE=us-central1-b

# Make bucket
GCS_DAG_BUCKET=$PROJECT_ID-datalab-airflow
gsutil mb gs://$GCS_DAG_BUCKET

DATALAB_PIPELINES_BUCKET=datalab-pipelines
AIRFLOW_VM_STARTUP_SCRIPT=gs://$DATALAB_PIPELINES_BUCKET/airflow_vm_startup.sh

# The default values here were grabbed from Pantheon's "command-line" button for creating a VM
gcloud beta compute --project $PROJECT_ID instances create $VM_NAME \
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
gcloud compute --project $PROJECT_ID ssh --zone $ZONE $VM_NAME
