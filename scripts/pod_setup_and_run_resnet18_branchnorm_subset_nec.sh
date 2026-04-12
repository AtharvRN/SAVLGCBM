#!/usr/bin/env bash
set -euo pipefail

cd /workspace/SAVLGCBM
bash scripts/pod_bootstrap_cbm.sh
bash scripts/pod_run_resnet18_branchnorm_subset_nec.sh
