#!/usr/bin/env bash
set -euo pipefail

ENV_YML="${CBM_CONDA_ENV_FILE:-envs/cbm.yml}"
ENV_NAME="${CBM_CONDA_ENV_NAME:-cbm}"

source /opt/conda/etc/profile.d/conda.sh

if [[ -f "$ENV_YML" ]]; then
  # Prefer a clean conda environment for reproducibility across pods. This avoids
  # polluting the base env and prevents pip from upgrading numpy/torch in ways
  # that conflict with preinstalled packages in the image.
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda env update -n "$ENV_NAME" -f "$ENV_YML" --prune
  else
    conda env create -n "$ENV_NAME" -f "$ENV_YML"
  fi
else
  # Backward-compatible fallback: clone base, then pip install requirements (legacy).
  if [[ ! -d "/opt/conda/envs/$ENV_NAME" ]]; then
    if conda env list | awk '{print $1}' | grep -qx "base"; then
      conda create -n "$ENV_NAME" --clone base -y || conda create -n "$ENV_NAME" python=3.10 -y
    else
      conda create -n "$ENV_NAME" python=3.10 -y
    fi
  fi
fi

conda activate "$ENV_NAME"

if [[ -n "${SAVLGCBM_ROOT:-}" ]]; then
  REPO_ROOT="$SAVLGCBM_ROOT"
elif [[ -d /scratch/SAVLGCBM ]]; then
  REPO_ROOT=/scratch/SAVLGCBM
elif [[ -d /workspace/SAVLGCBM ]]; then
  REPO_ROOT=/workspace/SAVLGCBM
else
  REPO_ROOT="$(pwd)"
fi

cd "$REPO_ROOT"

if [[ ! -f "$ENV_YML" ]]; then
  python -m pip install --upgrade pip
  python -m pip install setuptools wheel

  # If the caller set an on-PVC pip cache dir, ensure it exists.
  if [[ -n "${PIP_CACHE_DIR:-}" ]]; then
    mkdir -p "$PIP_CACHE_DIR"
  fi

  # Avoid accidentally pulling a brand new Torch/CUDA stack via pip (very large, slow,
  # and can mismatch the container CUDA). Install a pinned torch/torchvision first if
  # the env doesn't already have them, then install the rest of the requirements with
  # torch lines filtered out.
  if ! python - <<'PY'
import importlib
try:
    import torch, torchvision  # noqa: F401
    print("torch_ok", torch.__version__)
except Exception as e:
    print("torch_missing", repr(e))
    raise SystemExit(1)
PY
  then
    # Match the pytorch/pytorch:2.2.2-cuda12.1* base images.
    python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.2.2 torchvision==0.17.2
  fi

  REQ_TMP=/tmp/requirements_no_torch.txt

  # Keep pod bootstrap lean by default: notebooks are not needed for training/eval jobs
  # and pull in a large dependency tree. Set CBM_POD_INSTALL_NOTEBOOK=1 to include it.
  INSTALL_NOTEBOOK="${CBM_POD_INSTALL_NOTEBOOK:-0}"

  # Some workflows don't need the OpenAI client; it also pulls pydantic v2 which can
  # conflict with other stacks. Keep it opt-in for pod bootstrap.
  INSTALL_OPENAI="${CBM_POD_INSTALL_OPENAI:-0}"

  exclude_items=("torch" "torchvision")
  if [[ "$INSTALL_NOTEBOOK" != "1" ]]; then
    exclude_items+=("notebook")
  fi
  if [[ "$INSTALL_OPENAI" != "1" ]]; then
    exclude_items+=("openai")
  fi

  EXCLUDE_RE="^($(IFS='|'; echo "${exclude_items[*]}"))\\b"
  grep -vE "$EXCLUDE_RE" requirements.txt > "$REQ_TMP"
  python -m pip install -r "$REQ_TMP"
fi
