#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/SAVLGCBM"
PYTHON_BIN="/opt/conda/envs/cbm/bin/python"
ANNOTATIONS="$ROOT/annotations"
THRESHOLDS="-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2"

launch_eval_pipeline() {
  local gpu="$1"
  local ckpt_rel="$2"
  local nec_log="$3"
  local loc_log="$4"
  local loc_out="$5"
  local marker="$6"

  local ckpt_abs="$ROOT/$ckpt_rel"

  nohup /bin/bash -lc "
    set -euo pipefail
    cd '$ROOT'
    export PYTHONPATH='$ROOT'
    export CUDA_VISIBLE_DEVICES='$gpu'
    export PYTHONUNBUFFERED=1
    '$PYTHON_BIN' -u ./sparse_evaluation.py \
      --load_path '$ckpt_abs' \
      --annotation_dir '$ANNOTATIONS' \
      --lam 0.001 \
      --max_glm_steps 150 \
      --cbl_batch_size 256 \
      --num_workers 8 \
      --saga_batch_size 4096 \
      > '$nec_log' 2>&1
    '$PYTHON_BIN' -u ./scripts/evaluate_savlg_native_maps.py \
      --load_path '$ckpt_abs' \
      --annotation_dir '$ANNOTATIONS' \
      --output '$loc_out' \
      --eval_subset_mode gt_present \
      --map_normalization concept_zscore_minmax \
      --activation_thresholds='$THRESHOLDS' \
      --batch_size 512 \
      > '$loc_log' 2>&1
  " >/dev/null 2>&1 &

  echo "launched:$ckpt_abs" > "$marker"
}

maybe_launch_for_run() {
  local name="$1"
  local gpu="$2"
  local train_log="$3"
  local nec_log="$4"
  local loc_log="$5"
  local loc_out="$6"
  local marker="$7"

  [[ -f "$marker" ]] && return 0
  [[ -f "$train_log" ]] || return 0

  if ! grep -q "SAVLG-CBM test accuracy=" "$train_log"; then
    return 0
  fi

  local ckpt_rel
  ckpt_rel="$(grep "Saving SAVLG-CBM model to saved_models/cub/" "$train_log" | tail -n 1 | sed 's/.*Saving SAVLG-CBM model to //')"
  if [[ -z "$ckpt_rel" ]]; then
    echo "[$name] completed but checkpoint path could not be parsed" >&2
    return 0
  fi

  echo "[$name] training complete; launching NEC + localization on GPU $gpu"
  launch_eval_pipeline "$gpu" "$ckpt_rel" "$nec_log" "$loc_log" "$loc_out" "$marker"
}

AVG_TRAIN_LOG="$ROOT/logs/cub_savlg_r18_conv5_outside025_alpha010_dual_poolavg_b512.log"
AVG_NEC_LOG="$ROOT/logs/cub_savlg_r18_conv5_outside025_alpha010_dual_poolavg_b512_nec.log"
AVG_LOC_LOG="$ROOT/logs/native_savlg_r18_conv5_outside025_alpha010_dual_poolavg_b512_fulltest_sweep_m03_12.log"
AVG_LOC_OUT="$ROOT/results/native_savlg_r18_conv5_outside025_alpha010_dual_poolavg_b512_fulltest_sweep_m03_12.json"
AVG_MARKER="$ROOT/logs/.poolavg_eval_launched"

TOPK_TRAIN_LOG="$ROOT/logs/cub_savlg_r18_conv5_outside025_alpha010_dual_pooltopk_b512.log"
TOPK_NEC_LOG="$ROOT/logs/cub_savlg_r18_conv5_outside025_alpha010_dual_pooltopk_b512_nec.log"
TOPK_LOC_LOG="$ROOT/logs/native_savlg_r18_conv5_outside025_alpha010_dual_pooltopk_b512_fulltest_sweep_m03_12.log"
TOPK_LOC_OUT="$ROOT/results/native_savlg_r18_conv5_outside025_alpha010_dual_pooltopk_b512_fulltest_sweep_m03_12.json"
TOPK_MARKER="$ROOT/logs/.pooltopk_eval_launched"

echo "watcher:start $(date)"
while true; do
  maybe_launch_for_run "poolavg" 0 "$AVG_TRAIN_LOG" "$AVG_NEC_LOG" "$AVG_LOC_LOG" "$AVG_LOC_OUT" "$AVG_MARKER"
  maybe_launch_for_run "pooltopk" 1 "$TOPK_TRAIN_LOG" "$TOPK_NEC_LOG" "$TOPK_LOC_LOG" "$TOPK_LOC_OUT" "$TOPK_MARKER"
  sleep 120
done
