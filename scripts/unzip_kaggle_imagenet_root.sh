#!/usr/bin/env bash
set -euo pipefail

# Unzip the Kaggle ImageNet zip that lives at /root/imagenet-dataset into /root.
# Uses "unzip -n" so it can be re-run to resume without overwriting existing files.
#
# Usage:
#   bash scripts/unzip_kaggle_imagenet_root.sh
#
# Monitor:
#   tail -f /root/unzip_imagenet_dataset.log
#   du -sh /root/train /root/val /root/test 2>/dev/null

ZIP_PATH="${ZIP_PATH:-/root/imagenet-dataset}"
OUT_DIR="${OUT_DIR:-/root}"
LOG_PATH="${LOG_PATH:-/root/unzip_imagenet_dataset.log}"

if ! command -v unzip >/dev/null 2>&1; then
  echo "Missing dependency: unzip"
  echo "Install it (e.g. apt-get update && apt-get install -y unzip) and re-run."
  exit 1
fi

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Zip not found: $ZIP_PATH"
  exit 1
fi

mkdir -p "$OUT_DIR"
mkdir -p "$(dirname "$LOG_PATH")"

echo "Unzipping $ZIP_PATH -> $OUT_DIR (resume mode: -n) log=$LOG_PATH"

# -n: never overwrite existing files (safe resume)
# -q: quiet (log stays small); remove -q if you prefer file-by-file output.
unzip -n -q "$ZIP_PATH" -d "$OUT_DIR" >>"$LOG_PATH" 2>&1

echo "Unzip complete."

