#!/bin/bash

# Submission script that adds a job for model training.
#
# This script should be submitted from the root of this repository on HPG.
# It expects that a valid virtualenv has already been created with
# `poetry install`.

#SBATCH -J self_supervised_yolo_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=40gb
#SBATCH --mail-user=djpetti@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=self_supervised_yolo_train.%j.out    # Standard output log
#SBATCH --error=self_supervised_yolo_train.%j.err     # Standard error log
#SBATCH --account=cli2
#SBATCH --qos=cli2

set -e

source scripts/common.sh

# Prepare the environment.
prepare_environment

# Run the training.
export PYTHONPATH=${PYTHONPATH}:src/
poetry run yolo detect train model=data/01_raw/yolov8l.yml \
  epochs=100 \
  pretrained=yolov8l_ssl_moco_boll_1.5s.pt \
  batch=48 imgsz=640 cache=ram workers=8 \
  project=self_supervised name=yolo_val freeze=10 \
  data=data/05_model_input/boll_dataset/ground_dataset_small.yaml
