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

run_nec() {
  local gpu="$1"
  for i in "${!NAMES[@]}"; do
    local name="${NAMES[$i]}"
    local ckpt="${CKPTS[$i]}"
    local log="/workspace/SAVLGCBM/logs/resnet18_${name}_branchnorm_subset1000_nec.log"
    local csv="/workspace/SAVLGCBM/results/resnet18_${name}_branchnorm_subset1000_nec.csv"
    env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
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
}

run_loc() {
  local gpu="$1"
  for i in "${!NAMES[@]}"; do
    local name="${NAMES[$i]}"
    local ckpt="${CKPTS[$i]}"
    local log="/workspace/SAVLGCBM/logs/resnet18_${name}_subset1000_loc_t09.log"
    local out="/workspace/SAVLGCBM/results/resnet18_${name}_subset1000_loc_t09.json"
    env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/SAVLGCBM \
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
}

run_nec 0 &
PID_NEC=$!
run_loc 1 &
PID_LOC=$!

wait "${PID_NEC}"
wait "${PID_LOC}"

echo "DONE resnet18 branchnorm subset ablation"
