#!/bin/bash

# Submission script that adds a job for RotNet pretraining.
#
# This script should be submitted from the root of this repository on Sapelo.
# It expects that a valid virtualenv has already been created with
# `poetry install`.

#SBATCH --partition=patterli_p
#SBATCH -J cotton_mot_model_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --mem=20gb
#SBATCH --mail-user=daniel.petti@uga.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=cotton_rotnet_model_train.%j.out    # Standard output log
#SBATCH --error=cotton_rotnet_model_train.%j.err     # Standard error log

set -e

# Base directory we use for job output.
OUTPUT_BASE_DIR="/scratch/$(whoami)"
# Directory where our venv is located.
LARGE_FILES_DIR="/work/cylilab/cotton_mot"
# Directory where our data are located.
DATA_DIR="/scratch/$(whoami)/data"

function prepare_environment() {
  # Create the working directory for this job.
  job_dir="${OUTPUT_BASE_DIR}/job_${SLURM_JOB_ID}"
  mkdir "${job_dir}"
  echo "Job directory is ${job_dir}."

  # Copy the code.
  cp -Rd "${SLURM_SUBMIT_DIR}/"* "${job_dir}/"

  # Link to the input data directory and venv.
  rm -rf "${job_dir}/data"
  ln -s "${DATA_DIR}" "${job_dir}/data"
  ln -s "${LARGE_FILES_DIR}/.venv" "${job_dir}/.venv"

  # Create output directories.
  mkdir "${job_dir}/output_data"

  # Set the working directory correctly for Kedro.
  cd "${job_dir}"
}

# Prepare the environment.
prepare_environment

# Load needed modules.
ml Python/3.8.6-GCCcore-10.2.0
ml CUDA/11.3.1
ml cuDNN/8.1.0.77-CUDA-11.2.1

# Set this for deterministic runs. For more info, see
# https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
export PYTHONHASHSEED=0

# Run the training.
poetry run kedro run --pipeline=train_rotnet "$@"
