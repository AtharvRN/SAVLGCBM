# Final Technical Overview

This document is maintained during implementation and will be expanded as the repo evolves.

## Status

Initial planning and reference audit completed on 2026-03-31.

First implementation pass completed on 2026-03-31:

- VLG-CBM repo structure imported
- unified `model_name` dispatch added
- LF-CBM training path added
- compile-only verification completed

## Intended contents

This file will ultimately document:

1. final repo architecture
2. dataset handling and preprocessing
3. concept-bank handling
4. training flow per model
5. evaluation flow per model
6. sparse / NEC / ANEC logic
7. checkpoint conventions
8. deviations from upstream VLG-CBM
9. deviations from Medical_CBM
10. limitations and TODOs
11. reproducible commands

## Current architecture summary

- VLG-CBM is the base structure and baseline behavior reference.
- LF-CBM, SALF-CBM, and SAVLG-CBM are being integrated as separate model pipelines behind a unified registry.
- Sparse evaluation will remain VLG-style by default, with model-specific routing for LF/SALF/SAVLG.

## Current implementation state

### Implemented now

- shared training dispatch in `train_cbm.py`
- saved run metadata in `artifacts.json`
- LF-CBM training in `methods/lf.py`
- unified checkpoint evaluation in `eval_cbm.py`
- sparse-eval model routing in `sparse_evaluation.py`

### Not yet implemented

- SALF-CBM training port
- SAVLG-CBM training port
- spatial evaluation routing for SALF/SAVLG in this repo
- end-to-end runtime smoke tests

## Current known limitations

- SAVLG source code is available only in the local `Medical_CBM_dev_clean` snapshot, not the active `Medical_CBM` checkout.
- SALF/SAVLG on general-domain datasets require careful adaptation of spatial pseudo-label / annotation assumptions.
- Full training validation across all five datasets is out of scope for the first pass.
- Current LF sparse-eval compatibility is intended for linear LF concept layers; MLP LF checkpoints will need an additional loader path.
