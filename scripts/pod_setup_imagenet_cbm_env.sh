#!/usr/bin/env bash
set -euo pipefail

# One-shot environment setup for ImageNet CBM training inside a Nautilus pod.
#
# What it does:
# - Installs minimal OS deps (best-effort).
# - Bootstraps the conda env "cbm" using scripts/pod_bootstrap_cbm.sh
#   (clones base conda env so CUDA torch stays consistent).
# - Prints recommended environment variables for ImageNet training paths.
#
# Usage (inside pod):
#   bash /workspace/SAVLGCBM/scripts/pod_setup_imagenet_cbm_env.sh
#
# Optional overrides:
#   SAVLGCBM_ROOT=/scratch/SAVLGCBM bash .../pod_setup_imagenet_cbm_env.sh
#   IMAGENET_TRAIN_ROOT=... IMAGENET_VAL_ROOT=... IMAGENET_ANNOT_ROOT=... bash ...

maybe_apt_install() {
  if ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y >/dev/null 2>&1 || true
  apt-get install -y --no-install-recommends \
    git curl wget unzip jq ca-certificates \
    libgl1 libglib2.0-0 \
    >/dev/null 2>&1 || true
}

maybe_apt_install

if [[ -n "${SAVLGCBM_ROOT:-}" ]]; then
  ROOT="$SAVLGCBM_ROOT"
elif [[ -d /workspace/SAVLGCBM ]]; then
  ROOT=/workspace/SAVLGCBM
elif [[ -d /scratch/SAVLGCBM ]]; then
  ROOT=/scratch/SAVLGCBM
else
  echo "Could not find SAVLGCBM checkout at /workspace/SAVLGCBM or /scratch/SAVLGCBM."
  echo "Set SAVLGCBM_ROOT=/path/to/SAVLGCBM and re-run."
  exit 1
fi

cd "$ROOT"

# Shared pip cache (optional) helps speed up repeated pod setups.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/pip_cache}"
mkdir -p "$PIP_CACHE_DIR" || true

echo "[setup] Bootstrapping conda env: cbm"
bash scripts/pod_bootstrap_cbm.sh

echo
echo "[setup] Activate env:"
echo "  source /opt/conda/etc/profile.d/conda.sh && conda activate cbm"
echo

TRAIN_ROOT_DEFAULT="/scratch/imagenet_extract_runs_fresh/20260420_135422/train"
VAL_ROOT_DEFAULT="/scratch/imagenet_downloads/val"
ANNOT_ROOT_DEFAULT="/scratch/imagenet_annotations"
ACT_ROOT_DEFAULT="/scratch/savlgcbm_artifacts/saved_activations"
SAVE_ROOT_DEFAULT="/scratch/savlgcbm_artifacts/saved_models_imagenet"

echo "[setup] Recommended env vars (adjust if your paths differ):"
echo "  export IMAGENET_TRAIN_ROOT=\"${IMAGENET_TRAIN_ROOT:-$TRAIN_ROOT_DEFAULT}\""
echo "  export IMAGENET_VAL_ROOT=\"${IMAGENET_VAL_ROOT:-$VAL_ROOT_DEFAULT}\""
echo "  export IMAGENET_ANNOT_ROOT=\"${IMAGENET_ANNOT_ROOT:-$ANNOT_ROOT_DEFAULT}\""
echo "  export SAVLG_ACTIVATION_DIR=\"${SAVLG_ACTIVATION_DIR:-$ACT_ROOT_DEFAULT}\""
echo "  export SAVLG_SAVE_DIR=\"${SAVLG_SAVE_DIR:-$SAVE_ROOT_DEFAULT}\""
echo

echo "[setup] Quick import check:"
source /opt/conda/etc/profile.d/conda.sh
conda activate cbm
python - <<'PY'
import torch
import torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
import methods.savlg  # noqa: F401
print("SAVLG import OK")
PY

echo
echo "[setup] Example ImageNet SAVLG smoke command (adjust args):"
echo "  python -u train_cbm.py --config configs/unified/imagenet_savlg_cbm.json \\"
echo "    --model_name savlg_cbm --dataset imagenet --backbone resnet50 \\"
echo "    --annotation_dir \"$ANNOT_ROOT_DEFAULT\" --activation_dir \"$ACT_ROOT_DEFAULT\" \\"
echo "    --save_dir \"$SAVE_ROOT_DEFAULT\" --cbl_epochs 1 --cbl_batch_size 256"

