# Experiment Guide

This guide is being built incrementally.

## Intended matrix

For each dataset:

- CIFAR10
- CIFAR100
- CUB200
- Places365
- ImageNet

the repo will provide commands for:

- VLG-CBM training
- LF-CBM training
- SALF-CBM training
- SAVLG-CBM training
- full evaluation
- sparse / NEC evaluation

## Planned command shape

Training:

```bash
python train_cbm.py --config <config-path>
```

Evaluation:

```bash
python eval_cbm.py --load_path <run-dir>
```

Sparse / NEC:

```bash
python sparse_evaluation.py --load_path <run-dir>
```

## Current runnable commands

### VLG-CBM training

```bash
python train_cbm.py --config configs/unified/cifar10_vlg_cbm.json
python train_cbm.py --config configs/unified/cifar100_vlg_cbm.json
python train_cbm.py --config configs/unified/cub_vlg_cbm.json
python train_cbm.py --config configs/unified/places365_vlg_cbm.json
python train_cbm.py --config configs/unified/imagenet_vlg_cbm.json
```

### LF-CBM training

```bash
python train_cbm.py --config configs/unified/cifar10_lf_cbm.json
python train_cbm.py --config configs/unified/cifar100_lf_cbm.json
python train_cbm.py --config configs/unified/cub_lf_cbm.json
python train_cbm.py --config configs/unified/places365_lf_cbm.json
python train_cbm.py --config configs/unified/imagenet_lf_cbm.json
```

### Full evaluation

```bash
python eval_cbm.py --load_path <run_dir>
```

### Sparse / NEC evaluation

```bash
python sparse_evaluation.py --load_path <run_dir>
```

## Planned next commands

The following config examples exist but their training ports are still pending:

```bash
python train_cbm.py --config configs/unified/cifar10_salf_cbm.json
python train_cbm.py --config configs/unified/cifar10_savlg_cbm.json
```

The same pattern applies to CIFAR100, CUB, Places365, and ImageNet.

## Verification note

Only compilation checks have been run so far.
