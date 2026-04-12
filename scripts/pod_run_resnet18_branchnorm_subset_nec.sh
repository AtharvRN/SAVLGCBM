#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate cbm

cd /workspace/SAVLGCBM

MAX_IMAGES=1000

declare -a NAMES=(
  "conv45"
  "conv5"
  "conv4"
)

declare -a CKPTS=(
  "/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_23_45_15-1"
  "/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_08_02_43_40"
  "/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_08_02_43_40-1"
)

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  ckpt="${CKPTS[$i]}"
  log="/workspace/SAVLGCBM/logs/resnet18_${name}_branchnorm_subset1000_nec.log"
  csv="/workspace/SAVLGCBM/results/resnet18_${name}_branchnorm_subset1000_nec.csv"
  env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    /opt/conda/envs/cbm/bin/python sparse_evaluation.py \
      --load_path "${ckpt}" \
      --lam 0.001 \
      --max_glm_steps 150 \
      --cbl_batch_size 128 \
      --saga_batch_size 512 \
      --disable_activation_cache \
      --max_images "${MAX_IMAGES}" \
      --savlg_branch_norm_mode train_zscore \
      --result_file "${csv}" \
      >"${log}" 2>&1
done

echo "DONE resnet18 branchnorm subset NEC"
