#!/bin/bash

# Get current timestamp in format YYYYMMDD_HHMMSS
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Disable tokenizer parallelism to avoid warnings
export TOKENIZERS_PARALLELISM=false

accelerate launch --config_file config/train_dist.yaml train.py \
--train_meta_file dataset/FIGR-SVG-train.csv \
--val_meta_file dataset/FIGR-SVG-valid.csv \
--svg_folder dataset/FIGR-SVG-svgo \
--output_dir proj_log/ \
--project_name FIGR_SVG_${TIMESTAMP} \
--maxlen 512 \
--batchsize 20