#! /bin/bash

EXECUTION_DATE=$(date "+%Y%m%d-%H%M")
YEAR=$(date "+%Y")
MONTH=$(date "+%m")

PROJECT_DIR=$PWD
LOGS_DIR=${PROJECT_DIR}/logs/${YEAR}/${MONTH}

mkdir -p ${LOGS_DIR}

echo "================================== Start house price training ====================================="
papermill notebooks/e-commerce-text-classification-tf-idf.ipynb \
"${LOGS_DIR}/${EXECUTION_DATE}-e-commerce-text-classification-tf-idf-artifact.ipynb" \
-k python39 --report-mode --log-output --no-progress-bar

if [ $? != 0 ]; then
  echo "ERROR: failure during training!"
  exit 1
fi
echo "================================ SUCCESS: Done house price training ==================================="