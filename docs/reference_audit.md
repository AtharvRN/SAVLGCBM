# Reference Audit

## Scope

This document records the concrete local references inspected before code migration.
It is the basis for architectural decisions in this repo.

## Primary references

### VLG-CBM reference repo

- Local path: `/Users/atharvramesh/Projects/CBM/VLG-CBM`
- Role in this project:
  - base repository structure
  - dataset handling for general-domain datasets
  - config format and defaults
  - baseline VLG-CBM training flow
  - sparse / NEC evaluation flow
  - checkpoint and result conventions where practical
  - vendored local `glm_saga`

### Medical_CBM reference repo

- Local path: `/Users/atharvramesh/Projects/CBM/Medical_CBM`
- Role in this project:
  - LF-CBM behavior reference
  - SALF-CBM behavior reference
  - NEC extensions beyond upstream VLG-CBM
  - artifact naming patterns used by LF/SALF

### Medical_CBM dev-clean snapshot

- Local path: `/Users/atharvramesh/Projects/CBM/Medical_CBM_dev_clean`
- Role in this project:
  - SAVLG-CBM training implementation reference
  - `utils/vlg_core.py`
  - `utils/vlg_annotations.py`
- Why this is needed:
  - the active `Medical_CBM` checkout references `savlg_cbm.py`, `vlg_cbm.py`, and VLG utility files in `README.md`, but those files are not present in that working tree
  - the `dev_clean` snapshot contains `savlg_cbm.py`, `utils/vlg_core.py`, and `utils/vlg_annotations.py`

## Reusable VLG-CBM components

### Keep close to upstream

- `train_cbm.py`
  - top-level VLG training entrypoint
  - config loading pattern
  - save dir conventions
  - concept filtering, CBL training, final sparse layer training, metrics saving
- `data/utils.py`
  - general-domain dataset loading for CIFAR10, CIFAR100, CUB, Places365, ImageNet
  - backbone metadata and preprocessing
- `data/concept_dataset.py`
  - concept-annotation dataset wrapper
  - concept filtering
  - final-layer feature extraction path
- `model/cbm.py`
  - backbone wrappers
  - concept layer / normalization / final layer modules
  - SAGA integration
- `sparse_evaluation.py`
  - top-level sparse evaluation CLI
- `evaluations/sparse_utils.py`
  - NEC sweep implementation
  - truncated sparse weight export
- `configs/*.json`
  - dataset-specific defaults
- `glm_saga/`
  - must be vendored locally, not replaced with PyPI

### Notes on upstream assumptions

- Upstream VLG-CBM is single-model oriented.
- The data layout assumes annotation directories with `<dataset>_train` and `<dataset>_val` JSON shards.
- Checkpoints are directory-based and save `args.txt`, `concepts.txt`, concept layer weights, optional backbone weights, normalization stats, final-layer weights, and metrics.

## Reusable Medical_CBM components

### LF-CBM

- `label_free_cbm.py`
  - CLIP pseudo-label generation logic
  - projection-layer training against CLIP concept similarities
  - linear vs MLP concept bottleneck option
  - sparse final-layer training
  - artifact saving:
    - `W_c.pt` or `concept_layer.pt`
    - `W_g.pt`
    - `b_g.pt`
    - `proj_mean.pt`
    - `proj_std.pt`
    - `concepts.txt`
    - `train_metrics.json`
    - `val_metrics.json`
    - `method_log.json`

### SALF-CBM

- `spatial_aware_cbm.py`
  - spatial similarity tensor generation and caching
  - concept-map bottleneck training
  - pooled concept activations -> sparse final layer
  - artifact saving:
    - `concept_layer.pt`
    - `W_g.pt`
    - `b_g.pt`
    - `concept_mean.pt`
    - `concept_std.pt`
    - `config.json`
- `spatial_aware_eval.py`
  - SALF-specific evaluation and NEC support

### NEC extensions

- `evaluate_nec.py`
  - supports:
    - post-hoc NEC truncation
    - upstream-style GLM-SAGA NEC sweep
    - concept caching
    - labelwise vs global NEC
    - delegation to SALF evaluator for spatial checkpoints

## SAVLG-CBM reference components

### SAVLG training

- `Medical_CBM_dev_clean/savlg_cbm.py`
  - spatially-aware VLG pipeline using annotation-derived concept presence plus concept masks
  - pooled concept-map training with presence and mask supervision
  - sparse final-layer training

### VLG helper utilities

- `Medical_CBM_dev_clean/utils/vlg_core.py`
  - reusable concept-layer training helpers
  - final-layer dense/SAGA utilities
  - evaluation helpers
- `Medical_CBM_dev_clean/utils/vlg_annotations.py`
  - concept loading
  - annotation cache generation
  - annotation matrix loading
  - frequency-based concept filtering

## Key mismatches across references

### Dataset domain mismatch

- VLG-CBM uses general-domain datasets and torchvision-style loaders.
- Medical_CBM is mostly medical-dataset specific.
- Result:
  - keep VLG-CBM dataset layer as the canonical general-domain data stack
  - port only model logic and reusable helper logic from Medical_CBM

### Artifact naming mismatch

- VLG-CBM saves:
  - `args.txt`
  - `cbl.pt`
  - `train_concept_features_mean.pt`
  - `train_concept_features_std.pt`
  - `final.pt` via `FinalLayer.save_model`
- LF/SALF/SAVLG save:
  - `W_c.pt` or `concept_layer.pt`
  - `W_g.pt`, `b_g.pt`
  - `proj_mean.pt` / `proj_std.pt`
  - `concept_mean.pt` / `concept_std.pt`
  - `config.json`
  - `method_log.json`
- Result:
  - unify around explicit artifact metadata plus model-specific files
  - preserve VLG-compatible layout where reasonable
  - add compatibility loaders instead of forcing one naming scheme everywhere

### Model interface mismatch

- VLG-CBM concept bottleneck is global.
- SALF/SAVLG concept bottlenecks are spatial.
- LF-CBM can be linear or MLP and is trained against CLIP pseudo-label similarities.
- Result:
  - define a small model registry and explicit per-model pipeline interface
  - avoid over-abstracting training internals

## Initial migration conclusion

The safest strategy is:

1. Copy the VLG-CBM repository structure into this repo as the initial base.
2. Preserve the local VLG data and NEC flow with minimal renames.
3. Add a new shared layer for experiment config, model registry, artifact metadata, and model-specific pipeline modules.
4. Port LF-CBM first because it is global and structurally closest to VLG-CBM.
5. Port SALF-CBM next with separate spatial interfaces.
6. Port SAVLG-CBM last using the `dev_clean` reference.

## Open reference issues

- `Medical_CBM` active checkout does not currently contain `savlg_cbm.py` or VLG helper files referenced in its README.
- `Medical_CBM_dev_clean` contains `savlg_cbm.py` but not `vlg_cbm.py`.
- A CHEX-specific `vlg_cbm.py` exists elsewhere locally, but this repo will use upstream VLG-CBM itself as the VLG reference for general-domain datasets.
