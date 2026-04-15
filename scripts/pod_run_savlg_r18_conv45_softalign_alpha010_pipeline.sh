#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate cbm

cd /workspace/SAVLGCBM

GPU="${CUDA_VISIBLE_DEVICES:-0}"
TRAIN_LOG="/workspace/SAVLGCBM/logs/cub_savlg_r18_conv45_softalign_alpha010.log"
NEC_LOG="/workspace/SAVLGCBM/logs/cub_savlg_r18_conv45_softalign_alpha010_nec.log"
LOC_LOG="/workspace/SAVLGCBM/logs/native_savlg_r18_conv45_softalign_alpha010_gtpresent_zscore_t09.log"
LOC_OUT="/workspace/SAVLGCBM/results/native_savlg_r18_conv45_softalign_alpha010_gtpresent_zscore_t09.json"
VLG_WARM="/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42"

before_latest="$(ls -td /workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_* 2>/dev/null | head -n1 || true)"

env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
  /opt/conda/envs/cbm/bin/python train_cbm.py \
    --config configs/unified/cub_savlg_cbm.json \
    --annotation_dir /workspace/SAVLGCBM/annotations \
    --skip_train_val_eval \
    --cbl_epochs 75 \
    --cbl_batch_size 128 \
    --saga_batch_size 1024 \
    --num_workers 0 \
    --savlg_init_from_vlg_path "${VLG_WARM}" \
    --no_savlg_freeze_global_head \
    --savlg_local_loss_mode soft_align \
    --savlg_outside_penalty_w 0.0 \
    --loss_dice_w 0.0 \
    --savlg_residual_spatial_alpha 0.1 \
    --savlg_spatial_branch_mode multiscale_conv45 \
    --savlg_branch_arch dual \
    >"${TRAIN_LOG}" 2>&1

after_latest="$(ls -td /workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_* 2>/dev/null | head -n1 || true)"
if [[ -z "${after_latest}" || "${after_latest}" == "${before_latest}" ]]; then
  echo "ERROR: could not identify new conv45 alpha010 checkpoint" >&2
  exit 1
fi

env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
  /opt/conda/envs/cbm/bin/python -u sparse_evaluation.py \
    --load_path "${after_latest}" \
    --annotation_dir /workspace/SAVLGCBM/annotations \
    --lam 0.001 \
    --max_glm_steps 150 \
    >"${NEC_LOG}" 2>&1

env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/SAVLGCBM \
  /opt/conda/envs/cbm/bin/python scripts/evaluate_savlg_native_maps.py \
    --load_path "${after_latest}" \
    --annotation_dir /workspace/SAVLGCBM/annotations \
    --output "${LOC_OUT}" \
    --map_normalization concept_zscore_minmax \
    --activation_thresholds 0.9 \
    --box_iou_thresholds 0.1,0.3,0.5,0.7 \
    --eval_subset_mode gt_present \
    --batch_size 128 \
    >"${LOC_LOG}" 2>&1

echo "DONE checkpoint=${after_latest}"
