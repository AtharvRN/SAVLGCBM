#!/usr/bin/env bash
set -euo pipefail

# Pod-local ImageNet setup (Kaggle dataset): download + unzip into /root.
#
# Why /root?
# - On Nautilus pods, /root is node-local (fast) but EPHEMERAL (lost on pod restart).
# - This script is meant for quick throughput experiments, not long-term persistence.
#
# Prereqs:
# - Kaggle API token at ~/.kaggle/kaggle.json (expects {"username": "...", "key": "..."}).
#
# Usage:
#   bash scripts/pod2_prepare_kaggle_imagenet_root.sh
#
# Overrides:
#   DATASET_SLUG=mayurmadnani/imagenet-dataset OUT_DIR=/root bash ...
#
# Monitor:
#   tail -f /scratch/savlgcbm_artifacts/logs/kaggle_imagenet_prepare.log

DATASET_SLUG="${DATASET_SLUG:-mayurmadnani/imagenet-dataset}"
OUT_DIR="${OUT_DIR:-/root}"
LOG_PATH="${LOG_PATH:-/scratch/savlgcbm_artifacts/logs/kaggle_imagenet_prepare.log}"
LOCK_PATH="${LOCK_PATH:-/tmp/kaggle_imagenet_prepare.lock}"

mkdir -p "$(dirname "$LOG_PATH")"
mkdir -p "$OUT_DIR"

log() {
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG_PATH"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

ensure_kaggle() {
  if need_cmd kaggle; then
    return 0
  fi
  # Prefer pip install if missing.
  if need_cmd python && python -m pip --version >/dev/null 2>&1; then
    log "Installing kaggle CLI via pip (user install)"
    python -m pip install -q --user kaggle
    export PATH="$HOME/.local/bin:$PATH"
  fi
  need_cmd kaggle
}

ensure_creds() {
  local kaggle_json="${KAGGLE_JSON:-$HOME/.kaggle/kaggle.json}"
  if [[ ! -f "$kaggle_json" ]]; then
    log "ERROR: Kaggle credentials not found at $kaggle_json"
    log "Create it from https://www.kaggle.com/settings/account (API -> Create New Token)."
    exit 1
  fi
  # Kaggle CLI refuses insecure perms.
  chmod 600 "$kaggle_json" || true
}

download_zip_if_needed() {
  # Kaggle CLI downloads to <dataset>.zip by default.
  local zip_guess="${OUT_DIR}/$(basename "${DATASET_SLUG}").zip"
  if [[ -f "$zip_guess" ]]; then
    log "Found existing zip: $zip_guess"
    return 0
  fi

  # Try: if there is any large-ish zip already, reuse it.
  local any_zip
  any_zip="$(ls -1 "$OUT_DIR"/*.zip 2>/dev/null | head -n 1 || true)"
  if [[ -n "${any_zip:-}" ]]; then
    log "Found existing zip in OUT_DIR: $any_zip"
    return 0
  fi

  log "Downloading Kaggle dataset $DATASET_SLUG -> $OUT_DIR"
  # NOTE: kaggle CLI does not provide a stable resume; re-running re-downloads if partial.
  kaggle datasets download -d "$DATASET_SLUG" -p "$OUT_DIR" 2>&1 | tee -a "$LOG_PATH"
}

unzip_resume() {
  local any_zip
  any_zip="$(ls -1 "$OUT_DIR"/*.zip 2>/dev/null | head -n 1 || true)"
  if [[ -z "${any_zip:-}" ]]; then
    log "ERROR: no zip found in $OUT_DIR after download."
    exit 1
  fi

  log "Unzipping (resume mode: unzip -n) $any_zip -> $OUT_DIR"
  # -n never overwrites, so re-running safely resumes.
  unzip -n -q "$any_zip" -d "$OUT_DIR" >>"$LOG_PATH" 2>&1
  log "Unzip complete"
}

validate_layout() {
  local train_dir="$OUT_DIR/train"
  local val_dir="$OUT_DIR/val"
  local test_dir="$OUT_DIR/test"

  if [[ -d "$train_dir" ]]; then
    local n_classes
    n_classes="$(find "$train_dir" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
    log "train class dirs: $n_classes (expected 1000)"
  else
    log "WARN: missing $train_dir"
  fi

  if [[ -d "$val_dir" ]]; then
    local n_classes
    n_classes="$(find "$val_dir" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
    log "val class dirs: $n_classes (expected 1000)"
  else
    log "WARN: missing $val_dir"
  fi

  if [[ -d "$test_dir" ]]; then
    # test might be class-foldered or flat depending on the dataset; just report size.
    log "test present: $(du -sh "$test_dir" 2>/dev/null | awk '{print $1}')"
  else
    log "WARN: missing $test_dir"
  fi
}

main() {
  exec 9>"$LOCK_PATH"
  if ! flock -n 9; then
    log "Another prepare job is already running (lock: $LOCK_PATH)"
    exit 1
  fi

  log "Starting Kaggle ImageNet prepare: dataset=$DATASET_SLUG out_dir=$OUT_DIR"

  if ! need_cmd unzip; then
    log "ERROR: missing dependency: unzip"
    log "Install it (apt-get update && apt-get install -y unzip) and retry."
    exit 1
  fi

  if ! ensure_kaggle; then
    log "ERROR: kaggle CLI not found and could not be installed."
    exit 1
  fi
  ensure_creds

  download_zip_if_needed
  unzip_resume
  validate_layout

  log "Done."
}

main "$@"

