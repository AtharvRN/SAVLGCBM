#!/usr/bin/env bash
set -euo pipefail

# Minimal Kaggle ImageNet download + unzip into /root.
# Requires: ~/.kaggle/kaggle.json with a "key" field.

wget -c --progress=bar:force:noscroll \
  --header="Authorization: Bearer $(jq -r .key ~/.kaggle/kaggle.json)" \
  -O /root/imagenet-dataset.zip \
  "https://www.kaggle.com/api/v1/datasets/download/mayurmadnani/imagenet-dataset"

unzip -n -q /root/imagenet-dataset.zip -d /root

