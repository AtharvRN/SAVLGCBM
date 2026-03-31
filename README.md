# SAVLGCBM

Research codebase for training and evaluating multiple concept bottleneck models on the same general-domain setup used by VLG-CBM.

## Scope

Target model family:

1. `vlg_cbm`
2. `lf_cbm`
3. `salf_cbm`
4. `savlg_cbm`

Current implementation status:

- `vlg_cbm`: bootstrapped from the local upstream VLG-CBM repo
- `lf_cbm`: first unified training path added on top of the VLG dataset/backbone stack
- `salf_cbm`: planned, config examples added, training path not yet ported
- `savlg_cbm`: planned, config examples added, training path not yet ported

## Design summary

- The repo uses VLG-CBM as the base structure and baseline behavior reference.
- The local upstream `glm_saga` implementation is vendored directly.
- Multi-model support is introduced with a small shared layer under [`methods/`](./methods):
  - [`methods/common.py`](./methods/common.py)
  - [`methods/registry.py`](./methods/registry.py)
  - [`methods/lf.py`](./methods/lf.py)
- Sparse evaluation is routed by saved `model_name` metadata.

## Key entrypoints

- [`train_cbm.py`](./train_cbm.py): unified training entrypoint
- [`eval_cbm.py`](./eval_cbm.py): unified full-evaluation entrypoint
- [`sparse_evaluation.py`](./sparse_evaluation.py): sparse / NEC evaluation entrypoint

## Configs

Upstream VLG configs remain in [`configs/`](./configs).

Unified example configs live in [`configs/unified/`](./configs/unified):

- runnable now:
  - `*_vlg_cbm.json`
  - `*_lf_cbm.json`
- planning examples for next ports:
  - `*_salf_cbm.json`
  - `*_savlg_cbm.json`

## Documentation

Implementation docs are under [`docs/`](./docs):

- [`docs/reference_audit.md`](./docs/reference_audit.md)
- [`docs/architecture_plan.md`](./docs/architecture_plan.md)
- [`docs/progress_notes.md`](./docs/progress_notes.md)
- [`docs/final_technical_overview.md`](./docs/final_technical_overview.md)
- [`docs/experiment_guide.md`](./docs/experiment_guide.md)
- [`docs/model_notes/`](./docs/model_notes)

## Commands

Train upstream-style VLG-CBM:

```bash
python train_cbm.py --config configs/unified/cifar10_vlg_cbm.json
```

Train unified LF-CBM:

```bash
python train_cbm.py --config configs/unified/cifar10_lf_cbm.json
```

Evaluate a checkpoint:

```bash
python eval_cbm.py --load_path <run_dir>
```

Run sparse / NEC evaluation:

```bash
python sparse_evaluation.py --load_path <run_dir>
```

## Verification

Per user instruction, verification is currently limited to compilation checks only.

Completed check:

```bash
python -m compileall .
```

Runtime validation, dataset smoke tests, and training smoke tests were intentionally not run in this pass.
