#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate cbm

cd /workspace/SAVLGCBM

GPU="${CUDA_VISIBLE_DEVICES:-1}"
TRAIN_LOG="/workspace/SAVLGCBM/logs/cub_savlg_conv4_softalign_only.log"
NEC_LOG="/workspace/SAVLGCBM/logs/cub_savlg_conv4_softalign_only_nec_l1e3.log"
LOC_LOG="/workspace/SAVLGCBM/logs/native_savlg_r50_c4_softalign_only_gtpresent_zscore_t09.log"
LOC_OUT="/workspace/SAVLGCBM/results/native_savlg_r50_c4_softalign_only_gtpresent_zscore_t09.json"

before_latest="$(ls -td /workspace/SAVLGCBM/saved_models/savlg_cbm_cub_* 2>/dev/null | head -n1 || true)"

env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
  /opt/conda/envs/cbm/bin/python train_cbm.py \
    --config configs/unified/cub_savlg_cbm_vlgwarm_resnet50_cub_mm_residual_alpha020_softalign_outside025_conv5only_b32_v1.json \
    --annotation_dir /workspace/SAVLGCBM/annotations \
    --cbl_epochs 75 \
    --cbl_batch_size 256 \
    --saga_batch_size 1024 \
    --num_workers 0 \
    --no_savlg_freeze_global_head \
    --savlg_spatial_stage conv4 \
    --savlg_outside_penalty_w 0.0 \
    --loss_dice_w 0.0 \
    >"${TRAIN_LOG}" 2>&1

after_latest="$(ls -td /workspace/SAVLGCBM/saved_models/savlg_cbm_cub_* 2>/dev/null | head -n1 || true)"
if [[ -z "${after_latest}" || "${after_latest}" == "${before_latest}" ]]; then
  echo "ERROR: could not identify new conv4 soft-align checkpoint" >&2
  exit 1
fi

env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
  /opt/conda/envs/cbm/bin/python sparse_evaluation.py \
    --load_path "${after_latest}" \
    --lam 0.001 \
    --max_glm_steps 150 \
    --cbl_batch_size 128 \
    --saga_batch_size 512 \
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
