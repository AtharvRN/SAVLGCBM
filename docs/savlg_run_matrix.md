# SAVLG Run Matrix

This note consolidates the SAVLG run families that have actually been run and extracted so far.

The main reason for this file is to avoid mixing together:

- `resnet18_cub` vs `resnet50_cub_mm`
- `14 x 14` vs `7 x 7`
- `multiscale_conv45` vs older/weaker native-map families
- `coverage_w = 0` vs earlier nonzero-coverage runs
- `Adam` vs `SAM`

## Run Families

| Family | Backbone | Mask | Branch Mode | Alpha | Coverage | Optimizer | Status | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r18 oldbest alpha sweep` | `resnet18_cub` | `14 x 14` | `multiscale_conv45` | sweep `0..1` | default / unset | `Adam` | complete | [docs/savlg_alpha_sweep_r18_adam_only.md](/Users/atharvramesh/Projects/CBM/SAVLGCBM/docs/savlg_alpha_sweep_r18_adam_only.md) |
| `r18 oldbest Adam vs SAM` | `resnet18_cub` | `14 x 14` | `multiscale_conv45` | sweep `0..1` | default / unset | `Adam`, `SAM` | complete | [docs/savlg_alpha_sweep_r18_adam_vs_sam.md](/Users/atharvramesh/Projects/CBM/SAVLGCBM/docs/savlg_alpha_sweep_r18_adam_vs_sam.md) |
| `r18 exact cov0 sweep` | `resnet18_cub` | `14 x 14` | `multiscale_conv45` | sweep `0..1` | `0.0` | `Adam` | complete | [docs/savlg_r18_exact_cov0_alpha_sweep_summary.md](/Users/atharvramesh/Projects/CBM/SAVLGCBM/docs/savlg_r18_exact_cov0_alpha_sweep_summary.md) |
| `r50 exact earlier strong rerun` | `resnet50_cub_mm` | `7 x 7` | `multiscale_conv45` | `0.2` | `0.25` | `Adam` | complete | earlier reproduced checkpoint + JSON |
| `r50 exact relaunch` | `resnet50_cub_mm` | `7 x 7` | `multiscale_conv45` | `0.2` | `0.25` | `Adam` | failed | OOMKilled during feature caching |

## Verified Mask Sizes

These are the verified mask sizes for the main families:

- `resnet18_cub` oldbest / exact-cov0 family:
  - `mask_h = 14`
  - `mask_w = 14`
- `resnet50_cub_mm` exact-best family:
  - `mask_h = 7`
  - `mask_w = 7`

So the recent `resnet18` sweeps were **not** accidentally run with `7 x 7`.

## ResNet-18: Oldbest Adam Sweep

Recipe:

- backbone: `resnet18_cub`
- mask: `14 x 14`
- branch: `dual`
- spatial branch mode: `multiscale_conv45`
- optimizer: `Adam`
- alpha: sweep `0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0`

Best extracted metrics:

- best dense/full accuracy:
  - `alpha = 1.0`
  - `0.7630`
- best `AVGACC`:
  - `alpha = 0.6` and `1.0`
  - `0.7613`
- best `Acc@30`:
  - `alpha = 1.0`
  - `0.7618`

Reference:

- [docs/savlg_alpha_sweep_r18_adam_only.md](/Users/atharvramesh/Projects/CBM/SAVLGCBM/docs/savlg_alpha_sweep_r18_adam_only.md)

## ResNet-18: Oldbest Adam vs SAM

Same core `resnet18` / `14 x 14` / `multiscale_conv45` family, but compared across optimizers.

Best extracted metrics:

- Adam best dense:
  - `alpha = 1.0`
  - `0.7630`
- SAM best dense:
  - `alpha = 1.0`
  - `0.7627`
- Adam best `AVGACC`:
  - `0.7613`
- SAM best `AVGACC`:
  - `0.7615`

Read:

- optimizer differences are small
- alpha is the dominant variable

Reference:

- [docs/savlg_alpha_sweep_r18_adam_vs_sam.md](/Users/atharvramesh/Projects/CBM/SAVLGCBM/docs/savlg_alpha_sweep_r18_adam_vs_sam.md)

## ResNet-18: Exact `coverage_w = 0` Sweep

Recipe:

- backbone: `resnet18_cub`
- mask: `14 x 14`
- branch: `dual`
- spatial branch mode: `multiscale_conv45`
- optimizer: `Adam`
- `savlg_coverage_w = 0.0`
- alpha: sweep `0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0`

Best extracted metrics:

- best full accuracy:
  - `alpha = 1.0`
  - `0.7610`
- best `AVGACC`:
  - `alpha = 0.8`
  - `0.7605`
- best `Acc@30`:
  - `alpha = 1.0`
  - `0.7617`
- best native `mean IoU`:
  - `alpha = 0.6`
  - `0.2815`
- best native `mAP@0.3`:
  - `alpha = 0.0`
  - `0.4085`
- best native `mAP@0.5`:
  - `alpha = 0.0`
  - `0.1704`

Read:

- classification / NEC get better as `alpha` increases
- localization peaks earlier, then softens
- this is a real tradeoff

Reference:

- [docs/savlg_r18_exact_cov0_alpha_sweep_summary.md](/Users/atharvramesh/Projects/CBM/SAVLGCBM/docs/savlg_r18_exact_cov0_alpha_sweep_summary.md)

## ResNet-50: Earlier Strong Exact Rerun

Recipe verified from the earlier strong checkpoint:

- backbone: `resnet50_cub_mm`
- feature layer: `layer4`
- mask: `7 x 7`
- branch: `dual`
- spatial branch mode: `multiscale_conv45`
- `savlg_residual_spatial_alpha = 0.2`
- `savlg_outside_penalty_w = 0.25`
- `savlg_coverage_w = 0.25`
- `savlg_init_spatial_from_vlg = true`
- `savlg_freeze_global_head = false`
- `loss_mask_w = 0.25`
- `savlg_target_mode = soft_box`
- `savlg_local_loss_mode = soft_align`

Earlier extracted strong metrics:

- full accuracy:
  - `0.8573`
- native `mean IoU`:
  - `0.2822`
- native `mAP@0.3`:
  - `0.4467`
- native `mAP@0.5`:
  - `0.2152`

This is the strong `resnet50` reference point that later comparisons should be anchored to.

## ResNet-50: Exact Relaunch Status

The recent exact `resnet50_cub_mm` relaunch was corrected to match the earlier strong recipe, but it still failed.

What happened:

- status:
  - `OOMKilled`
- reason:
  - container RAM limit hit
- it was **not** a GPU OOM
- it died after:
  - `SAVLG feature cache (train): 100%`
- so the failure is host RAM pressure during or just after feature caching

Current takeaway:

- the recipe alignment is now correct
- the relaunch still needs more pod memory headroom

## Practical Summary

If the question is “which combinations are real and trustworthy right now?”:

- trustworthy `r18` family:
  - the `14 x 14` `multiscale_conv45` sweeps
- trustworthy `r50` reference:
  - the earlier exact `7 x 7` strong rerun with `alpha = 0.2`, `outside = 0.25`, `coverage = 0.25`

If the question is “what should not be mixed together?”:

- `r18 14 x 14` vs `r50 7 x 7`
- `coverage_w = 0` vs `coverage_w = 0.25`
- old weaker localization sweeps vs the exact strong reruns
