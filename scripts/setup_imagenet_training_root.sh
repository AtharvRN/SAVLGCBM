#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <root> <train_tar> <val_tar> <devkit_dir>"
  exit 1
fi

ROOT="$1"
TRAIN_TAR="$2"
VAL_TAR="$3"
DEVKIT_DIR="$4"

python scripts/build_imagenet_torchvision_root.py \
  --root "$ROOT" \
  --train-archive "$TRAIN_TAR" \
  --val-archive "$VAL_TAR" \
  --devkit-dir "$DEVKIT_DIR"

python scripts/audit_imagenet_tree.py \
  --train-root "$ROOT/train" \
  --devkit-dir "$DEVKIT_DIR"
