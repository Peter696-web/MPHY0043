#!/bin/bash
set -e

# 1) Create labels
python src/prepare/new_label.py

# 2) Generate split manifest (ratio: 60:10:10)
python src/prepare/make_split.py --output data/split_manifest.json

# 3) Extract features according to the manifest (e.g., base model)
python src/prepare/feature_extraction.py \
	--dataset cholec80 \
	--data_root ./cholec80 \
	--output_dir ./data/features \
	--models base \
	--split_manifest data/split_manifest.json

# 4) Align the features and labels to processed folder
python src/prepare/preprocess.py

echo "Preprocessing finished."
