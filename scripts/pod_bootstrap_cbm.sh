#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d /opt/conda/envs/cbm ]]; then
  conda create -n cbm python=3.10 -y
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate cbm

cd /workspace/SAVLGCBM

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install \
  pandas==2.3.3 \
  numpy==1.26.4 \
  opencv-python==4.11.0.86 \
  mmengine==0.10.7 \
  mmcv-lite==2.2.0 \
  mmpretrain==1.2.0 \
  tensorboard \
  ftfy \
  regex \
  pytorchcv
