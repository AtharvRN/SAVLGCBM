# Architecture And Implementation Plan

## Goals

Build a pragmatic research codebase that trains and evaluates four CBM variants on the same general-domain setup used by VLG-CBM:

1. VLG-CBM
2. LF-CBM
3. SALF-CBM
4. SAVLG-CBM

The codebase should keep VLG-CBM recognizable while making fair cross-model comparison easy.

## Non-goals

- Do not rewrite the entire repository into a generic framework.
- Do not replace VLG-CBM sparse evaluation with a new protocol.
- Do not force medical-dataset abstractions onto CIFAR/CUB/Places/ImageNet.
- Do not depend on a PyPI `glm_saga` variant that may differ from the local implementation.

## Proposed target structure

```text
SAVLGCBM/
  configs/
    base/
    cifar10/
    cifar100/
    cub/
    places365/
    imagenet/
  concept_files/
  data/
  datasets/
  evaluations/
  glm_saga/
  models/
    registry.py
    common.py
    vlg.py
    lf.py
    salf.py
    savlg.py
  pipelines/
    common.py
    vlg_pipeline.py
    lf_pipeline.py
    salf_pipeline.py
    savlg_pipeline.py
  scripts/
  docs/
  train_cbm.py
  eval_cbm.py
  sparse_evaluation.py
  README.md
```

## Design principles

### 1. Keep VLG-CBM as the visible spine

The repo should still look like VLG-CBM to a reader familiar with the original code:

- same dataset folders
- same concept files
- same `configs/`
- same `train_cbm.py` and `sparse_evaluation.py` entrypoints where possible
- same local `glm_saga`

### 2. Add the minimum abstraction layer needed for four models

The abstraction should answer only these questions:

- which model variant is selected
- how the concept bottleneck is trained
- how concept features are extracted for final-layer fitting
- how a checkpoint is saved / loaded
- how NEC evaluation should interpret that checkpoint

Anything beyond that is unnecessary for the first pass.

### 3. Separate global and spatial CBMs

There are effectively two families:

- global bottleneck models:
  - VLG-CBM
  - LF-CBM
- spatial bottleneck models:
  - SALF-CBM
  - SAVLG-CBM

The code should not pretend these are identical internally.

### 4. Preserve checkpoint compatibility where reasonable

Target behavior:

- VLG checkpoints remain close to upstream VLG-CBM.
- LF/SALF/SAVLG checkpoints keep their explicit weight files.
- each run also writes a shared metadata file describing:
  - `model_name`
  - dataset
  - backbone
  - concepts path
  - normalization artifact names
  - final-layer artifact names
  - evaluation style

## Minimal shared abstractions

### Model registry

One registry mapping `model_name` to a pipeline implementation:

- `vlg_cbm`
- `lf_cbm`
- `salf_cbm`
- `savlg_cbm`

Responsibilities:

- validate config compatibility
- construct model components
- train bottleneck
- extract normalized concept features
- fit/load final layer
- save/load checkpoint metadata

### Shared artifact metadata

Add a small metadata file, likely `run_config.json` or `artifacts.json`, with:

- `model_name`
- `dataset`
- `backbone`
- `concepts_file`
- `concept_layer_format`
- `normalization_format`
- `final_layer_format`
- `supports_spatial_nec`
- `sparse_eval_style`

This avoids brittle checkpoint introspection.

### Shared concept feature interface

All models should provide one callable path to obtain concept features for final-layer fitting:

- global models return `N x C`
- spatial models return:
  - pooled global concepts `N x C` for classifier training
  - optionally spatial maps `N x C x H x W` for spatial evaluation

### Shared NEC interface

Common NEC flow:

1. load checkpoint
2. extract or load cached concept features
3. refit / sweep sparse final-layer weights with local `glm_saga`
4. truncate weights to requested NEC budgets
5. evaluate using the same dataset split and metric conventions
6. save:
   - path summary
   - NEC metrics csv/json
   - truncated weights

Model-specific branching:

- VLG and LF can use the global concept-feature path directly
- SALF and SAVLG may also expose spatial evaluator hooks where needed

## Dataset integration plan

Canonical dataset layer remains VLG-CBM `data/utils.py` + `data/concept_dataset.py`.

### Required datasets

- CIFAR10
- CIFAR100
- CUB200
- Places365
- ImageNet

### Dataset plan

- Keep VLG-CBM dataset naming:
  - `cifar10_train`, `cifar10_val`
  - `cifar100_train`, `cifar100_val`
  - `cub_train`, `cub_val`
  - `places365_train`, `places365_val`
  - `imagenet_train`, `imagenet_val`
- Preserve `DATASET_FOLDER` environment variable behavior.
- Preserve annotation directory naming convention:
  - `<annotation_dir>/<dataset>_train/<idx>.json`
  - `<annotation_dir>/<dataset>_val/<idx>.json`

### New requirement for LF/SALF/SAVLG on general-domain data

The dataset layer must support:

- raw images with class labels only
- optional concept annotation access for VLG/SAVLG
- optional spatial annotation access for SAVLG
- optional CLIP pseudo-label generation for LF/SALF

This means the dataset code should stay domain-agnostic; the model pipeline decides whether to consume:

- class labels only
- concept vectors
- concept maps

## Concept bank handling plan

### Reuse VLG concept assets directly

- keep `concept_files/*.txt`
- keep `*_per_class.json`
- keep filtered concept lists

### Add concept loading helpers

Provide shared helpers for:

- flat text concept banks
- per-class JSON concept banks
- loading filtered concept lists
- writing run-local `concepts.txt`

### Filtering behavior

- VLG-CBM:
  - preserve annotation-based concept filtering behavior
- LF-CBM:
  - preserve CLIP-score and interpretability filtering behavior as separate options
- SALF-CBM:
  - preserve spatial similarity based filtering behavior
- SAVLG-CBM:
  - preserve annotation frequency / confidence based filtering behavior

## Training flow by model

### VLG-CBM

Keep closest to upstream:

1. load dataset and annotations
2. filter concepts from annotation evidence
3. train concept bottleneck layer
4. cache normalized concept features
5. fit sparse final layer with SAGA
6. save metrics and artifacts

### LF-CBM

Port from Medical_CBM:

1. load concept bank
2. load backbone features
3. compute CLIP image-text similarities as pseudo-label targets
4. train projection layer to match CLIP concept similarities
5. normalize projected concepts
6. fit sparse final layer with SAGA
7. save LF-specific artifacts and method log

### SALF-CBM

Port from Medical_CBM:

1. compute or load spatial CLIP target tensor
2. train spatial concept map head
3. pool concept maps into global concept activations
4. normalize pooled concepts
5. fit sparse final layer with SAGA
6. save spatial artifacts and config

### SAVLG-CBM

Port from `Medical_CBM_dev_clean/savlg_cbm.py`:

1. load concept presence + box/mask supervision from annotation JSONs
2. train spatial concept map head with global presence and spatial mask loss
3. pool concept maps into global concept activations
4. normalize pooled concepts
5. fit sparse final layer with SAGA
6. save spatial artifacts and config

## Sparse / NEC integration plan

### Baseline rule

VLG-CBM sparse evaluation remains the canonical reference.

### First-pass implementation

- keep VLG-style `sparse_evaluation.py`
- extend it to inspect `model_name`
- route to:
  - global NEC evaluator for VLG and LF
  - spatial NEC evaluator for SALF and SAVLG when needed

### Evaluation outputs

Standardize on:

- `metrics.csv` or `nec_metrics.csv`
- `upstream_nec_metrics.csv` when upstream-style sweep is used
- `W_g@NEC=<k>.pt`
- `b_g@NEC=<k>.pt`
- cached concept tensors for repeat sweeps

### NEC styles to support

At minimum:

- upstream-style GLM-SAGA sweep from VLG-CBM
- post-hoc truncation where already implemented in Medical_CBM

## Config plan

### Compatibility strategy

- preserve the original VLG config keys
- add a small number of new keys for multi-model support:
  - `model_name`
  - `clip_name` for LF/SALF
  - spatial head options for SALF/SAVLG
  - `sparse_eval_style`

### Config examples to add

For each dataset:

- `configs/<dataset>/vlg_cbm.json`
- `configs/<dataset>/lf_cbm.json`
- `configs/<dataset>/salf_cbm.json`
- `configs/<dataset>/savlg_cbm.json`

Not all spatial configs will be fully runnable immediately if spatial supervision assets are absent; document this explicitly.

## Checkpoint / output convention plan

Each run directory should contain:

- `config.json` or `args.txt`
- `artifacts.json`
- `concepts.txt`
- concept bottleneck weights
- normalization stats
- final-layer weights
- train/val/test metrics
- sparse / NEC outputs
- logs where applicable

### Preferred naming

- keep existing VLG names for VLG
- keep existing LF/SALF/SAVLG names for the model-specific artifacts
- add `artifacts.json` so loaders can stay simple

## Deviations already planned

### From upstream VLG-CBM

- add a model registry
- add explicit artifact metadata
- generalize train/eval entrypoints to support four CBMs
- preserve VLG sparse logic but extend it to other models

### From Medical_CBM

- remove medical-dataset coupling from LF/SALF/SAVLG ports
- replace CheXpert/COVID dataset loaders with VLG general-domain loaders
- adapt concept supervision assumptions to VLG-style annotation layout

## Step-by-step implementation plan

1. Write docs and lock the target structure.
2. Copy the VLG-CBM repository structure into this repo.
3. Vendor the VLG `glm_saga` directory unchanged.
4. Add `models/registry.py` and a shared config loader.
5. Refactor the upstream VLG entrypoint just enough to dispatch `model_name=vlg_cbm`.
6. Port LF-CBM as the first non-VLG model using shared dataset/backbone utilities.
7. Extend sparse evaluation to understand LF checkpoints.
8. Port SALF-CBM with separate spatial concept interfaces.
9. Port SAVLG-CBM using `Medical_CBM_dev_clean/savlg_cbm.py` as the training reference.
10. Add dataset/model config examples and experiment docs.
11. Run lightweight CLI validation and import checks.

## First-pass deliverable for this round

This round should accomplish:

1. repo bootstrap from VLG-CBM
2. documentation scaffold
3. unified registry/config skeleton
4. VLG pipeline wired through the unified entrypoint
5. LF-CBM added next as the first ported non-VLG model

SALF and SAVLG can be scaffolded after the shared interfaces are in place.

## Risks and constraints

- SALF/SAVLG rely on spatial concept supervision assets not present in the upstream VLG dataset pipeline.
- ImageNet and Places365 runs are large; only smoke validation is realistic locally.
- Some Medical_CBM code assumes medical labels and metrics; those parts must not be ported blindly.
- `Medical_CBM` active checkout is missing some files referenced by its own README; use `Medical_CBM_dev_clean` selectively and document this provenance.

## Open questions

- Do we want exact upstream CLI parity for `train_cbm.py`, or allow `--model_name` as a required extension?
- Should LF use the same backbone wrappers as VLG or keep its own encoder abstraction?
- For SALF on general-domain datasets, should the first pass support prompt-grid spatial pseudo-labels only, patch-token mode only, or both?
- For SAVLG on general-domain datasets, do we require box-level annotations generated in VLG format, or introduce a dedicated spatial annotation schema?
