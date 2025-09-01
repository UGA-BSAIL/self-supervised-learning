#!/bin/bash

# Submission script that adds a job for model training.
#
# This script should be submitted from the root of this repository on Sapelo.
# It expects that a valid virtualenv has already been created with
# `poetry install`.

#SBATCH --partition=hpg-b200
#SBATCH -J self_supervised_model_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=56gb
#SBATCH --account=cli2
#SBATCH --qos=cli2
#SBATCH --mail-user=djpetti@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=self_supervised_model_train.%j.out    # Standard output log
#SBATCH --error=self_supervised_model_train.%j.err     # Standard error log

set -e

source scripts/common.sh

# Prepare the environment.
prepare_environment
copy_data_to_scratch

# Run the training.
poetry run kedro run --pipeline=train_simclr --env=a100 "$@"
