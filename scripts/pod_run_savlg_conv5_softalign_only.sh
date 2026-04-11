#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate cbm

cd /workspace/SAVLGCBM

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONUNBUFFERED=1 \
/opt/conda/envs/cbm/bin/python train_cbm.py \
  --config configs/unified/cub_savlg_cbm_vlgwarm_resnet50_cub_mm_residual_alpha020_softalign_outside025_conv5only_b32_v1.json \
  --cbl_epochs 75 \
  --cbl_batch_size 512 \
  --saga_batch_size 1024 \
  --num_workers 0 \
  --no_savlg_freeze_global_head \
  --savlg_outside_penalty_w 0.0 \
  --savlg_coverage_w 0.0 \
  --loss_dice_w 0.0
