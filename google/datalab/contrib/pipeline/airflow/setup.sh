#!/usr/bin/env bash
PROJECT=${1:-cloud-ml-dev}
EMAIL=${2:-rajivpb@google.com}
ZONE=${3:-us-central1}
ENVIRONMENT=${3:-rajivpb-airflow}

gcloud config set project $PROJECT
gcloud config set account $EMAIL
gcloud auth login --activate $EMAIL

# We use the default cluster spec.
gcloud container clusters create $ENVIRONMENT --zone $ZONE

# Deploying the airflow container
kubectl run airflow --image=gcr.io/cloud-airflow-releaser/airflow-worker-scheduler-1.8
