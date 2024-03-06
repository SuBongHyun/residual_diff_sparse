#!/bin/bash

# 1. environment setting
conda create -n residual_diff python=3.7
conda activate residual_diff
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt


# 2. model checkpoints
CHECKPOINT_DIR=./checkpoints
mkdir -p "$CHECKPOINT_DIR/AAPM_256_ncsnpp_continuous"
