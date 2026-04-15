#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d /opt/conda/envs/cbm ]]; then
  conda create -n cbm python=3.10 -y
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate cbm

cd /workspace/SAVLGCBM

python -m pip install --upgrade pip
python -m pip install setuptools wheel
python -m pip install -r requirements.txt
