#!/usr/bin/env bash
set -euo pipefail

cd /workspace/SAVLGCBM

bash scripts/pod_bootstrap_cbm.sh
bash scripts/pod_run_savlg_r18_conv45_softalign_alpha010_pipeline.sh
