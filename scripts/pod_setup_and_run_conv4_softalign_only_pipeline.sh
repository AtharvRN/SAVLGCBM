#!/usr/bin/env bash
set -euo pipefail

cd /workspace/SAVLGCBM

bash scripts/pod_bootstrap_cbm.sh
bash scripts/pod_run_savlg_conv4_softalign_only_pipeline.sh
