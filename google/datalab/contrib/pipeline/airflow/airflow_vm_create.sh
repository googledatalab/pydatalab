#!/usr/bin/env bash
PROJECT_ID=${1:-cloud-ml-dev}
VM_NAME=${2:-instance-30}
ZONE=${3:-us-central1-b}

# Make a GCS bucket. This will fail if the bucket already exists, and that's ok.
GCS_DAG_BUCKET=$PROJECT_ID-datalab-airflow
gsutil mb gs://$GCS_DAG_BUCKET

# Create the VM.
AIRFLOW_VM_STARTUP_SCRIPT=gs://datalab-pipelines/airflow_vm_startup.sh
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

# Meditate.
sleep 90

# TODO(rajivpb): To be deleted; left here only for convenience
gcloud compute --project $PROJECT_ID ssh --zone $ZONE $VM_NAME
