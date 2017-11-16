#!/usr/bin/env bash
VM_NAME="instance-7"
PROJECT_ID="cloud-ml-dev"
ZONE="us-central1-b"

# TODO(rajivpb): These should have more official values
AIRFLOW_VM_STARTUP_SCRIPT="gs://rajivpb-airflow-testing/airflow_vm_startup.sh"
GCS_DAG_BUCKET_FOR_AIRFLOW="rajivpb-airflow-testing"
GCS_DAG_PATH_FOR_AIRFLOW="/dags"

# The default values here were grabbed from Pantheon's "command-line" button for creating a VM
gcloud beta compute --project $PROJECT_ID instances create $VM_NAME \
--zone $ZONE \
--machine-type "n1-standard-1" \
--network "default" \
--maintenance-policy "MIGRATE" \
--service-account "236417448818-compute@developer.gserviceaccount.com" \
--scopes "https://www.googleapis.com/auth/cloud-platform" \
--min-cpu-platform "Automatic" \
--image "debian-9-stretch-v20171025" \
--image-project "debian-cloud" \
--boot-disk-size "10" \
--boot-disk-type "pd-standard" \
--boot-disk-device-name $VM_NAME \
--metadata-from-file startup-script-url=$AIRFLOW_VM_STARTUP_SCRIPT \

# TODO(rajivpb): Eventually, we need to set this metadata so that they can be parameters
# --metadata GCS_DAG_BUCKET=$GCS_DAG_BUCKET_FOR_AIRFLOW,GCS_DAG_PATH=$GCS_DAG_PATH_FOR_AIRFLOW \

gcloud compute --project $PROJECT_ID ssh --zone $ZONE $VM_NAME
