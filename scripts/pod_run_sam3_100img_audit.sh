#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate cbm

cd /workspace/SAVLGCBM
mkdir -p logs

LOG="/workspace/SAVLGCBM/logs/cub_sam3_100img_3concept_train.log"
MANIFEST="saved_activations/sam3_concept_masks/cub_medsam3_concept_masks_100img_3concept_v1/train/manifest.json"

DATASET_FOLDER=/workspace/SAVLGCBM/datasets \
python scripts/generate_sam3_concept_masks.py \
  --config configs/sam3/cub_concept_masks_medsam3_100img_3concept_pod.json \
  --split train \
  --run \
  --max_images 100 \
  --max_concepts 3 \
  --overwrite \
  2>&1 | tee "${LOG}"

python scripts/build_sam3_concept_mask_audit.py \
  --manifest "${MANIFEST}" \
  --max_records 300

python - <<'PY'
import json
from collections import Counter
from pathlib import Path

p = Path("saved_activations/sam3_concept_masks/cub_medsam3_concept_masks_100img_3concept_v1/train/manifest.json")
m = json.loads(p.read_text())
print("manifest", p)
print("statuses", dict(Counter(r.get("status") for r in m["records"])))
print("records", m["record_count"])
print("audit", p.parent / "audit" / "index.html")
PY
