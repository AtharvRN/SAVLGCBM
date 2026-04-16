#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate cbm

if [ -d /workspace/SAVLGCBM-sam3-run ]; then
  REPO_DIR="${REPO_DIR:-/workspace/SAVLGCBM-sam3-run}"
else
  REPO_DIR="${REPO_DIR:-/workspace/SAVLGCBM}"
fi

cd "${REPO_DIR}"
mkdir -p logs

LOG="${REPO_DIR}/logs/cub_sam3_20img_audit_train.log"
MANIFEST="saved_activations/sam3_concept_masks/cub_medsam3_concept_masks_20img_audit_v1/train/manifest.json"
if [ -d "${REPO_DIR}/datasets/CUB" ]; then
  DATASET_DIR="${DATASET_DIR:-${REPO_DIR}/datasets}"
else
  DATASET_DIR="${DATASET_DIR:-/workspace/SAVLGCBM/datasets}"
fi

DATASET_FOLDER="${DATASET_DIR}" \
python scripts/generate_sam3_concept_masks.py \
  --config configs/sam3/cub_concept_masks_medsam3_20img_audit_pod.json \
  --split train \
  --run \
  --max_images 20 \
  --max_concepts 3 \
  --overwrite \
  2>&1 | tee "${LOG}"

python scripts/build_sam3_concept_mask_audit.py \
  --manifest "${MANIFEST}" \
  --max_records 60

python - <<'PY'
import json
from collections import Counter
from pathlib import Path

p = Path("saved_activations/sam3_concept_masks/cub_medsam3_concept_masks_20img_audit_v1/train/manifest.json")
m = json.loads(p.read_text())
print("manifest", p)
print("statuses", dict(Counter(r.get("status") for r in m["records"])))
print("records", m["record_count"])
print("audit", p.parent / "audit" / "index.html")
PY
