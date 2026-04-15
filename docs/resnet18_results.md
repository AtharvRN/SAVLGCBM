# ResNet18 Results

This file collects the completed `resnet18_cub` results that were previously spread across [cub_results.md](/Users/atharvramesh/Projects/CBM/SAVLGCBM/docs/cub_results.md), plus the newer soft-align-only retrains and NEC reruns.

## Stable Cross-Method Baselines

These are the corrected / promoted `resnet18_cub` baselines.

| Method | Checkpoint | Dense Acc | ACC@5 | AVGACC | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `VLG-CBM` | upstream-style `resnet18_cub` | `0.7594` | `0.7546` | `0.7556` | Reference ceiling on `resnet18_cub` |
| `LF-CBM` | `lf_cbm_cub_2026_04_03_04_22_18` | `0.7402` | `0.5147` | `0.6817` | Original LF bank + fixed LF similarity |
| `SALF-CBM` | `salf_cbm_cub_2026_04_03_05_30_10` | `0.7320` | `0.5335` | `0.6826` | Original LF bank + fixed LF similarity |
| `SAVLG-CBM` | `savlg_cbm_cub_2026_04_02_20_07_04` | `0.7459` | `0.6917` | `0.7170` | First clean SAVLG baseline |

## Earlier SAVLG Follow-Ups

| Variant | Checkpoint | Dense Acc | ACC@5 | AVGACC | Localization Summary |
| --- | --- | ---: | ---: | ---: | --- |
| `Dice-enabled local loss` | `savlg_cbm_cub_2026_04_02_21_27_12` | `0.7456` | `0.6926` | `0.7083` | Slightly worse sparse than baseline |
| `Multiscale dual-branch + local-MIL` | `savlg_cbm_cub_2026_04_05_22_26_28` | `0.7506` | `0.7446` | `0.7480` | Native IoU `0.1194`, `mAP@0.5 = 0.4915` |

## Warm-Start Frozen-Global Local-Loss Ablation

All six dense runs landed at `0.7591`, and all six NEC reruns landed at `ACC@5 = 0.7549`, `AVGACC = 0.7560`.

| Variant | Dense Acc | ACC@5 | AVGACC | Mean IoU | mAP@0.3 | mAP@0.5 | mAP@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `containment only` | `0.7591` | `0.7549` | `0.7560` | `0.1202` | `0.0435` | `0.0325` | `0.0245` |
| `local-mil only` | `0.7591` | `0.7549` | `0.7560` | `0.0632` | `0.0247` | `0.0230` | `0.0222` |
| `soft-align only` | `0.7591` | `0.7549` | `0.7560` | `0.1241` | `0.0472` | `0.0326` | `0.0235` |
| `soft-align + local-mil` | `0.7591` | `0.7549` | `0.7560` | `0.1250` | `0.0297` | `0.0252` | `0.0223` |
| `soft-align + outside` | `0.7591` | `0.7549` | `0.7560` | `0.1246` | `0.0491` | `0.0336` | `0.0237` |
| `soft-align + local-mil + outside` | `0.7591` | `0.7549` | `0.7560` | `0.1255` | `0.0311` | `0.0258` | `0.0224` |

## Residual Spatial Coupling Sprint

| Variant | Dense Acc | ACC@5 | ACC@10 | ACC@15 | ACC@20 | ACC@25 | ACC@30 | AVGACC | Mean IoU | mAP@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Residual-Alpha-0.05` | `0.7592` | `0.7560` | `0.7568` | `0.7568` | `0.7570` | `0.7570` | `0.7572` | `0.7568` | `0.1309` | `0.0656` |
| `Residual-Alpha-0.10` | `0.7585` | `0.7572` | `0.7584` | `0.7580` | `0.7579` | `0.7584` | `0.7585` | `0.7581` | `0.1307` | `0.0657` |
| `Residual-Alpha-0.05-PlusMIL` | `0.7591` | `0.7563` | `0.7572` | `0.7572` | `0.7573` | `0.7584` | `0.7584` | `0.7575` | `0.1216` | `0.0434` |

## Spatial-Source Residual Ablation

This holds the residual recipe fixed and changes only the spatial source.

| Variant | Dense Acc | ACC@5 | AVGACC | Mean IoU | mAP@0.3 | mAP@0.5 | mAP@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `conv4+conv5` | `0.7592` | `0.7560` | `0.7568` | baseline residual localizer | baseline residual localizer | baseline residual localizer | baseline residual localizer |
| `conv5-only` | `0.7592` | `0.7560` | `0.7567` | `0.1251` | `0.1168` | `0.0630` | `0.0312` |
| `conv4-only` | `0.7587` | `0.7547` | `0.7559` | `0.1445` | `0.1170` | `0.0710` | `0.0395` |
| `conv3-only` | `0.7587` | `0.7549` | `0.7559` | `0.1113` | `0.0906` | `0.0488` | `0.0323` |

Note:
- The `conv4+conv5` row above matches the `Residual-Alpha-0.05` multiscale residual baseline and is already represented in the residual table.

## Residual Alpha Sweep

This sweep used `conv5-only` with `soft-align + outside`.

| Alpha | Dense Acc | AVGACC | Notes |
| --- | ---: | ---: | --- |
| `0.00` | `0.7591` | `0.7556` | pure global anchor |
| `0.02` | `0.7589` | `0.7561` | slight sparse improvement |
| `0.10` | `0.7592` | `0.7585` | best sparse tier |
| `0.20` | `0.7606` | `0.7585` | best overall tradeoff |
| `0.30` | `0.7606` | `0.7580` | dense-best tier |
| `0.40` | `0.7594` | `0.7585` | best sparse tier |

## Recent ResNet18 Soft-Align-Only Retrains

These are the newer retrains that removed outside penalty and Dice and used the simplified soft-align-only setup.

Common settings:
- backbone: `resnet18_cub`
- warm start from `VLG`
- `savlg_local_loss_mode = soft_align`
- `savlg_outside_penalty_w = 0.0`
- `loss_dice_w = 0.0`
- fused-logit BCE still active
- no branch normalization before fusion

### Dense Results

| Variant | Checkpoint | Alpha | Dense Acc | Status |
| --- | --- | ---: | ---: | --- |
| `conv4` | `savlg_cbm_cub_2026_04_11_21_34_50-1` | `0.05` | `0.5682` | complete |
| `conv5` | `savlg_cbm_cub_2026_04_11_21_43_56` | `0.05` | `0.7534` | complete |
| `conv4+conv5` | `savlg_cbm_cub_2026_04_11_23_43_32` | `0.05` | `0.7541` | complete |
| `conv4+conv5` | `savlg_cbm_cub_2026_04_12_06_30_05` | `0.10` | `0.7579` | complete |
| `conv4+conv5` | `savlg_cbm_cub_2026_04_12_06_43_49` | `0.20` | interrupted | no usable checkpoint artifacts |
| `conv4+conv5` | `alpha=0.20, batch_size=256` rerun | `0.20` | `0.7558` | complete |
| `conv4+conv5` | `alpha=0.30, batch_size=256` rerun | `0.30` | `0.7556` | complete |
| `SALF-CBM` | `salf_cbm_cub_2026_04_12_00_43_57` | n/a | `0.7318` | complete |

### NEC Results

| Variant | ACC@5 | ACC@10 | ACC@15 | ACC@20 | ACC@25 | ACC@30 | AVGACC | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `conv5` | `0.7038` | `0.7382` | `0.7425` | `0.7453` | `0.7461` | `0.7473` | `0.7372` | complete |
| `conv4+conv5` | `0.7038` | `0.7382` | `0.7425` | `0.7453` | `0.7461` | `0.7473` | `0.7372` | complete |
| `conv4` | pending | pending | `0.4392` | `0.4834` | `0.5086` | `0.5166` | `0.4207` | complete |
| `conv4+conv5 (alpha=0.10)` | pending | pending | pending | pending | `0.7470` | `0.7480` | `0.7354` | complete |
| `conv4+conv5 (alpha=0.30)` | pending | pending | pending | `0.7440` | `0.7454` | `0.7472` | `0.7372` | complete |
| `SALF-CBM` | pending | pending | pending | pending | pending | pending | pending | not run in this new cycle |

Note:
- The recent `conv4+conv5` NEC log currently reports the same endpoint metrics as the recent `conv5` NEC run. This is preserved here exactly as observed from the log and should be rechecked if that parity looks suspicious.

## Recent Localization Results

These are the recent native-map localization evaluations from the newer soft-align-only line.

| Variant | Split / Sweep | Best Threshold(s) | Mean IoU | mAP@0.1 | mAP@0.3 | mAP@0.5 | mAP@0.7 | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `conv5` | subset `1000`, fixed `0.9` | `0.9` | `0.02038` | `0.06622` | `0.01130` | `0.00217` | `0.000155` | complete |
| `conv5` | subset `1000`, sweep `0.05..0.95` | IoU / `mAP@0.3`: `0.05`; `mAP@0.5`: `0.65`; `mAP@0.7`: `0.70` | `0.05505` | n/a | `0.02955` | `0.00464` | `0.000774` | complete |
| `conv4+conv5` | subset `1000`, fixed `0.9` | `0.9` | `0.01866` | `0.06499` | `0.00975` | `0.00139` | `0.000464` | complete |
| `conv4+conv5` | subset `1000`, sweep `-0.3..1.2` | IoU / `mAP@0.3`: `-0.3`; `mAP@0.5`: `0.3`; `mAP@0.7`: `0.5` | `0.05505` | n/a | `0.02955` | `0.00387` | `0.000464` | complete |
| `conv4+conv5 (alpha=0.10)` | full test, sweep `-0.3..1.2` | IoU / `mAP@0.3`: `-0.3`; `mAP@0.5`: `0.3`; `mAP@0.7`: `0.5` | `0.05435` | `0.19443` | `0.02805` | `0.00345` | `0.000529` | complete |

## Residual Pooling Ablation

This holds the strong old-style `conv5` residual recipe fixed:
- `savlg_branch_arch=dual`
- `savlg_spatial_branch_mode=multiscale_conv45`
- `savlg_spatial_stage=conv5`
- `savlg_outside_penalty_w=0.25`
- `savlg_residual_spatial_alpha=0.1`

### Dense + NEC

| Pooling | Checkpoint | Dense Acc | ACC@5 | ACC@10 | ACC@15 | ACC@20 | ACC@25 | ACC@30 | AVGACC | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `lse` | `savlg_cbm_cub_2026_04_12_13_45_48` | `0.7591` | `0.7577` | `0.7585` | `0.7585` | `0.7587` | `0.7587` | `0.7589` | `0.7585` | complete |
| `avg` | `savlg_cbm_cub_2026_04_13_03_05_45-1` | `0.7606` | `0.7561` | `0.7570` | `0.7577` | `0.7579` | `0.7584` | `0.7584` | `0.7576` | complete |
| `topk (0.2)` | `savlg_cbm_cub_2026_04_13_03_44_56` | `0.7613` | `0.7566` | `0.7573` | `0.7582` | `0.7585` | `0.7582` | `0.7582` | `0.7579` | complete |

### Localization

Full-test native-map sweep, `thresholds = -0.3 .. 1.2`, `batch_size = 512`, `map_normalization = concept_zscore_minmax`.

| Pooling | Best Mean IoU | Best mAP@0.1 | Best mAP@0.3 | Best mAP@0.5 | Best mAP@0.7 | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `avg` | `thr=-0.3, 0.05435` | `thr=-0.3, 0.19443` | `thr=-0.3, 0.02805` | `thr=-0.3, 0.00331` | `thr=-0.3, 0.000473` | complete |
| `topk (0.2)` | `thr=-0.3, 0.05435` | `thr=-0.3, 0.19443` | `thr=-0.3, 0.02805` | `thr=-0.3, 0.00331` | `thr=-0.3, 0.000473` | complete |

## Dual Conv45 Activation-Threshold Diagnostic

This is a small diagnostic run on the strong dual `conv4+conv5` checkpoint:
- checkpoint: `savlg_cbm_cub_2026_04_07_23_45_15-1`
- subset size: `10` images
- threshold source: `normalized_map`
- normalization: `concept_zscore_minmax`
- threshold sweep: `-1.0 .. 1.0`

Note:
- because `concept_zscore_minmax` rescales to `[0,1]`, thresholds below `0` are equivalent and produce identical outputs.

| Metric | Best Value | Threshold |
| --- | ---: | ---: |
| `mean_iou` | `0.1088853` | `0.5` |
| `mask_iou` | `0.1225064` | `0.4` |
| `dice` | `0.1661743` | `0.4` |
| `mAP@0.1` | `0.3000` | `-1.0` |
| `mAP@0.3` | `0.1200` | `0.4` |
| `mAP@0.5` | `0.0600` | `0.3` |
| `mAP@0.7` | `0.0600` | `0.4` |
| `point_hit` | `0.1200` | see per-threshold JSON |
| `soft_iou` | `0.0765871` | distribution metric |
| `mass_in_box` | `0.1252879` | distribution metric |

Result files:
- `native_savlg_conv45_old_actmap_tneg1_1_subset10_rerun.json`
- `native_savlg_conv45_old_actmap_tneg1_1_subset10_rerun_dice.json`

## Source

Primary older source:
- [cub_results.md](/Users/atharvramesh/Projects/CBM/SAVLGCBM/docs/cub_results.md)

This file is intended as the compact `resnet18`-only view of that larger document, plus the new April 11–12 retrains and NEC reruns.
