#!/usr/bin/env bash
set -euo pipefail

# Downloads the Kaggle dataset zip via the Kaggle API endpoint and unzips it.
#
# Requires: ~/.kaggle/kaggle.json containing {"username": "...", "key": "..."}
#
# Example:
#   bash scripts/download_kaggle_imagenet.sh /root/imagenet_kaggle
#
# Notes:
# - This uses the bearer token pattern you provided.
# - Uses wget resume (-c) and a readable progress bar.

URL="https://www.kaggle.com/api/v1/datasets/download/mayurmadnani/imagenet-dataset"
ZIP_PATH="/root/dataset.zip"
OUT_DIR="${1:-/root/imagenet_kaggle}"
KAGGLE_JSON="${KAGGLE_JSON:-$HOME/.kaggle/kaggle.json}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    return 1
  fi
}

maybe_install_deps() {
  # Best-effort; do not fail the script if apt isn't available.
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y >/dev/null 2>&1 || true
    apt-get install -y --no-install-recommends jq wget unzip ca-certificates >/dev/null 2>&1 || true
  fi
}

maybe_install_deps

for c in jq wget unzip; do
  if ! need_cmd "$c"; then
    echo "Missing dependency: $c"
    echo "Install it (e.g. apt-get install -y $c) and re-run."
    exit 1
  fi
done

if [[ ! -f "$KAGGLE_JSON" ]]; then
  echo "Kaggle credentials not found at: $KAGGLE_JSON"
  echo "Expected ~/.kaggle/kaggle.json with a 'key' field."
  exit 1
fi

KAGGLE_KEY="$(jq -r '.key // empty' "$KAGGLE_JSON")"
if [[ -z "$KAGGLE_KEY" || "$KAGGLE_KEY" == "null" ]]; then
  echo "Could not read .key from $KAGGLE_JSON"
  exit 1
fi

mkdir -p "$(dirname "$ZIP_PATH")"
mkdir -p "$OUT_DIR"

echo "Downloading to $ZIP_PATH"
wget -c --progress=bar:force:noscroll \
  --header="Authorization: Bearer ${KAGGLE_KEY}" \
  -O "$ZIP_PATH" \
  "$URL"

echo "Unzipping into $OUT_DIR"
unzip -q -o "$ZIP_PATH" -d "$OUT_DIR"

echo "Done."

