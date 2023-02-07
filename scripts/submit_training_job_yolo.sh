#!/bin/bash

# Submission script that adds a job for model training.
#
# This script should be submitted from the root of this repository on HPG.
# It expects that a valid virtualenv has already been created with
# `poetry install`.

#SBATCH --partition=gpu
#SBATCH -J self_supervised_yolo_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=40gb
#SBATCH --mail-user=djpetti@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=self_supervised_yolo_train.%j.out    # Standard output log
#SBATCH --error=self_supervised_yolo_train.%j.err     # Standard error log

set -e

# Base directory we use for job output.
OUTPUT_BASE_DIR="/blue/cli2/$(whoami)/job_scratch/"
# Directory where our data and venv are located.
LARGE_FILES_DIR="/blue/cli2/$(whoami)/ssl/"

function prepare_environment() {
  # Create the working directory for this job.
  job_dir="${OUTPUT_BASE_DIR}/job_${SLURM_JOB_ID}"
  mkdir "${job_dir}"
  echo "Job directory is ${job_dir}."

  # Copy the code.
  cp -Rd "${SLURM_SUBMIT_DIR}/"* "${job_dir}/"

  # Link to the input data directory and venv.
  rm -rf "${job_dir}/data"
  ln -s "${LARGE_FILES_DIR}/data" "${job_dir}/data"
  ln -s "${LARGE_FILES_DIR}/.venv" "${job_dir}/.venv"

  # Create output directories.
  mkdir "${job_dir}/output_data"
  mkdir "${job_dir}/logs"

  # Set the working directory correctly for Kedro.
  cd "${job_dir}"
}

# Prepare the environment.
prepare_environment

source scripts/load_common.sh

# Run the training.
poetry run python -m yolov5.train --cfg data/01_raw/yolov5_l.yml --epochs 100 \
  --batch-size 128 --imgsz 640 --cache ram --workers 8 \
  --project self_supervised --name yolo_val --entity djpetti \
  --weights "" --freeze 10 \
  --data data/05_model_input/flower_dataset/dataset_quarter.yaml
