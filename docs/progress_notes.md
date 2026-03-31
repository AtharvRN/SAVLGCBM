# Progress Notes

## 2026-03-31

### Initial inspection

- Confirmed target repo started nearly empty.
- Confirmed local references exist:
  - `/Users/atharvramesh/Projects/CBM/VLG-CBM`
  - `/Users/atharvramesh/Projects/CBM/Medical_CBM`
  - `/Users/atharvramesh/Projects/CBM/Medical_CBM_dev_clean`

### VLG-CBM findings

- Upstream structure is compact and suitable as the base.
- Key reusable components identified:
  - `train_cbm.py`
  - `data/utils.py`
  - `data/concept_dataset.py`
  - `model/cbm.py`
  - `evaluations/sparse_utils.py`
  - `sparse_evaluation.py`
  - `configs/*.json`
  - local `glm_saga/`

### Medical_CBM findings

- LF-CBM and SALF-CBM implementations are present in the active checkout.
- NEC evaluator is richer than upstream VLG and already supports:
  - upstream-style GLM-SAGA sweep
  - post-hoc NEC
  - concept caching
  - SALF delegation
- The active `Medical_CBM` checkout does not contain:
  - `savlg_cbm.py`
  - `vlg_cbm.py`
  - `utils/vlg_core.py`
  - `utils/vlg_annotations.py`

### Medical_CBM dev-clean findings

- `Medical_CBM_dev_clean` contains:
  - `savlg_cbm.py`
  - `utils/vlg_core.py`
  - `utils/vlg_annotations.py`
- This snapshot will be used as the SAVLG reference where the active repo is incomplete.

### Architecture decision

- Base the new codebase on VLG-CBM structure and evaluation behavior.
- Add the smallest possible shared layer:
  - config normalization
  - model registry
  - artifact metadata
  - per-model pipeline modules
- Port order:
  1. VLG-CBM
  2. LF-CBM
  3. SALF-CBM
  4. SAVLG-CBM

### Documentation created

- `docs/reference_audit.md`
- `docs/architecture_plan.md`
- `docs/progress_notes.md`
- `docs/final_technical_overview.md`
- `docs/model_notes/`
- `docs/experiment_guide.md`

### Immediate next steps

1. Bootstrap this repo by copying the VLG-CBM code structure.
2. Preserve local `glm_saga`.
3. Add unified registry/config plumbing.
4. Rewire VLG training through the new registry without changing behavior.
5. Port LF-CBM next.

### First implementation pass completed

- Copied the upstream VLG-CBM code structure into this repo.
- Added a small shared method layer:
  - `methods/common.py`
  - `methods/registry.py`
  - `methods/lf.py`
- Updated `train_cbm.py` to dispatch by `model_name`.
- Added `artifacts.json` metadata writing for saved runs.
- Added `eval_cbm.py` as a unified checkpoint evaluator for current runnable models.
- Updated `sparse_evaluation.py` to route by saved `model_name`.
- Added unified config examples under `configs/unified/`.

### Verification boundary

- Compilation check completed successfully with:
  - `python -m compileall /Users/atharvramesh/Projects/CBM/SAVLGCBM`
- Runtime validation was intentionally skipped per user instruction.
