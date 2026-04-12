#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate cbm

cd /workspace/SAVLGCBM

ANN_DIR="/workspace/SAVLGCBM/annotations"
MAX_IMAGES=1000
ACT_THR="0.9"
BOX_IOUS="0.1,0.3,0.5,0.7"

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
  log="/workspace/SAVLGCBM/logs/resnet18_${name}_subset1000_loc_t09.log"
  out="/workspace/SAVLGCBM/results/resnet18_${name}_subset1000_loc_t09.json"
  env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/SAVLGCBM \
    /opt/conda/envs/cbm/bin/python scripts/evaluate_savlg_native_maps.py \
      --load_path "${ckpt}" \
      --annotation_dir "${ANN_DIR}" \
      --output "${out}" \
      --map_normalization concept_zscore_minmax \
      --activation_thresholds "${ACT_THR}" \
      --box_iou_thresholds "${BOX_IOUS}" \
      --eval_subset_mode gt_present \
      --batch_size 128 \
      --max_images "${MAX_IMAGES}" \
      >"${log}" 2>&1
done

echo "DONE resnet18 branchnorm subset localization"
