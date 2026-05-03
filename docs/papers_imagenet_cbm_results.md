# Paper ImageNet CBM Results

This note is a compact paper-facing summary of the current ImageNet results across the CBM variants we have evaluated so far.

The structure is:

1. sparse classification results for all CBMs
2. localization results

## Scope and Caveats

- All SAVLG results come from the completed ImageNet runs and follow-up evals in this workspace.
- SALF-CBM results come from the original-author pretrained ImageNet checkpoint.
- VLG-CBM results in this note are currently **localization-only**. We do not yet have a consolidated VLG-CBM sparse classification eval table in this workspace.
- These are **checkpoint-native** comparisons:
  - VLG-CBM used its own concept set (`4300`)
  - SALF-CBM used its own concept set (`4741`)
  - SAVLG used the concept set bundled with each SAVLG artifact
- So this is not a controlled same-vocabulary comparison.

## Sparse Classification Results

### Main sparse classification summary

| Model | Best raw sparse top-1 | Best raw sparse top-5 | Best exact `NEC=5` top-1 | Best exact `NEC=30` top-1 | `AVGACC` over NEC {5,10,15,20,25,30} | Notes |
|---|---:|---:|---:|---:|---:|---|
| SAVLG | `0.76926` | `0.9312` | `0.76304` | `0.79140` | `0.78578` | best raw sparse model comes from `mask_w=1.0`; best low-NEC result comes from the new `b128` scratch-scale checkpoint with `glm_sweep_lammax2e3` |
| SALF-CBM pretrained | n/a separate raw sparse run not recorded | n/a | `0.53534` | `0.75366` | `0.69784` | original-author pretrained checkpoint, evaluated with exact per-class NEC truncation |
| VLG-CBM pretrained | n/a | n/a | n/a | n/a | n/a | sparse classification not yet consolidated in this workspace |

### SAVLG sparse classification details

Best raw sparse SAVLG model on full 50k val:

| Setting | Top-1 | Top-5 | NNZ |
|---|---:|---:|---:|
| `mask_w=1.0`, `lambda=0.0003` | `0.76926` | `0.9312` | `71,594` |

Best current exact-truncation SAVLG NEC result:

- run:
  - `savlg_imagenet_full_7ep_a100_alpha02_scratch_scale_b128_w32_pf2_maskw1_20260502T182559Z_savlg-imagenet-a100-7ep-scratch-scale-b128-w32-fm7xr`
- artifact:
  - `glm_sweep_lammax2e3`
- 50k val-tar eval:
  - `/workspace/savlg_imagenet_standalone_runs/.../glm_sweep_lammax2e3/nec_eval_test.json`
- saved NEC weights:
  - `/workspace/savlg_imagenet_standalone_runs/.../glm_sweep_lammax2e3/W_g@NEC=*.pt`
  - `/workspace/savlg_imagenet_standalone_runs/.../glm_sweep_lammax2e3/b_g@NEC=*.pt`

| NEC | NNZ | Top-1 | Top-5 |
|---:|---:|---:|---:|
| 5 | `5,402` | `0.76304` | `0.92562` |
| 10 | `10,078` | `0.78884` | `0.94004` |
| 15 | `15,451` | `0.78984` | `0.94108` |
| 20 | `22,811` | `0.79058` | `0.94176` |
| 25 | `25,647` | `0.79100` | `0.94206` |
| 30 | `30,065` | `0.79140` | `0.94238` |

Summary:

- `ACC@NEC=5`: `0.76304`
- `AVGACC`: `0.78578` over NEC `{5,10,15,20,25,30}`
- Average top-5 over the same NEC set: `0.93882`

Additional completed 10ep scratch `mask_w=1.0` exact-truncation NEC result:

- run:
  - `savlg_imagenet_full_10ep_a100_alpha02_scratch_maskw1_20260502T103652Z_savlg-imagenet-full-a100-10ep-alpha02-scratch-maskw1-cnnm4`
- artifact:
  - `glm_one_lambda_lam2e4`
- persistent path:
  - `/workspace/savlg_imagenet_standalone_runs/.../glm_one_lambda_lam2e4`

| NEC | NNZ | Top-1 | Top-5 |
|---:|---:|---:|---:|
| 5 | `5,294` | `0.76256` | `0.93098` |
| 10 | `10,406` | `0.78688` | `0.94264` |
| 15 | `15,338` | `0.78750` | `0.94372` |
| 20 | `21,127` | `0.78780` | `0.94390` |
| 25 | `26,773` | `0.78870` | `0.94436` |
| 30 | `31,242` | `0.78868` | `0.94430` |

Summary:

- `ACC@NEC=5`: `0.76256`
- `AVGACC`: `0.78369` over NEC `{5,10,15,20,25,30}`
- Average top-5 over the same NEC set: `0.94165`

Previous best exact-truncation SAVLG NEC result:

| NEC | Top-1 | Top-5 |
|---:|---:|---:|
| 5 | `0.47636` | `0.73036` |
| 10 | `0.65124` | `0.85918` |
| 15 | `0.69702` | `0.89070` |
| 20 | `0.72080` | `0.90518` |
| 25 | `0.73522` | `0.91260` |
| 30 | `0.74322` | `0.91704` |

Additional exact-truncation result for the `mask_w=1.0` sparse checkpoint on 50k val:

| NEC | Top-1 | Top-5 |
|---:|---:|---:|
| 5 | `0.44848` | `0.69330` |
| 10 | `0.61948` | `0.83742` |
| 15 | `0.68408` | `0.88384` |
| 20 | `0.71458` | `0.90250` |
| 25 | `0.73392` | `0.91308` |
| 30 | `0.74518` | `0.91966` |

Interpretation:

- SAVLG is strong in dense and moderately sparse regimes.
- The new `b128` scratch-scale checkpoint substantially improves exact low-NEC classification.
- This new run is now stronger than the current SALF-CBM pretrained NEC reference in this workspace.
- `alpha=0.2` remains better than the earlier `alpha=0.8` reference at low NEC.

### SALF-CBM pretrained sparse classification details

Original-author pretrained ImageNet SALF-CBM checkpoint:

| NEC | Top-1 | Top-5 |
|---:|---:|---:|
| 5 | `0.53534` | `0.80406` |
| 10 | `0.67546` | `0.87864` |
| 15 | `0.72668` | `0.90076` |
| 20 | `0.74510` | `0.90986` |
| 25 | `0.75078` | `0.91412` |
| 30 | `0.75366` | `0.91526` |

Dense SALF-CBM reference:

| Model | Top-1 | Top-5 |
|---|---:|---:|
| SALF-CBM pretrained dense | `0.75564` | `0.91632` |

Interpretation:

- The new SAVLG exact-NEC run is stronger than this SALF-CBM reference on both `ACC@NEC=5` and `AVGACC`.

### VLG-CBM sparse classification status

We do not yet have a consolidated VLG-CBM sparse classification eval in this workspace summary.

So for VLG-CBM, the current paper-facing ImageNet evidence is:

- localization comparison is available
- sparse classification table is not yet available here

## Localization Results

### Headline localization metrics

We report:

- `LocAcc@0.3`: localization accuracy at IoU `0.3`
- `LocAcc@0.5`: localization accuracy at IoU `0.5`
- `MeanIoU`: mask IoU at mean-threshold heatmap

### Dense / native map comparison

| Model | LocAcc@0.3 | LocAcc@0.5 | MeanIoU |
|---|---:|---:|---:|
| VLG-CBM Grad-CAM | `0.5246` | `0.3572` | `0.2405` |
| SALF-CBM native | `0.5985` | `0.4375` | `0.2691` |
| SAVLG native | `0.6542` | `0.4873` | `0.3449` |

Interpretation:

- SAVLG dense/native localization is already better than both VLG-CBM Grad-CAM and SALF-CBM on the current ImageNet localization protocol.

### Best current sparse SAVLG localization result

Best saved sparse SAVLG checkpoint:

| Model | LocAcc@0.3 | LocAcc@0.5 | MeanIoU |
|---|---:|---:|---:|
| SAVLG sparse best current | `0.90894` | `0.60047` | `0.42558` |

Other metrics for that same sparse SAVLG checkpoint:

| Metric | Value |
|---|---:|
| `soft_iou` | `0.40754` |
| `mass_in_gt` | `0.59783` |
| `point_hit` | `0.75024` |
| `mask_iou@0.5` | `0.45055` |

Interpretation:

- This is the strongest localization result we currently have for ImageNet SAVLG.
- Under the current evaluator, it beats the earlier SALF-CBM localization baseline and is much stronger than VLG-CBM Grad-CAM.

## Current Paper-Level Summary

### Classification

- Best dense SAVLG classification is already strong.
- Best raw sparse SAVLG classification is also strong at moderate NNZ.
- Low-NEC SAVLG still trails pretrained SALF-CBM.
- VLG-CBM sparse classification is not yet consolidated in this workspace note.

### Localization

- Best current localization ranking:
  1. SAVLG sparse
  2. SAVLG dense/native
  3. SALF-CBM native
  4. VLG-CBM Grad-CAM

So the strongest current ImageNet story for SAVLG is:

- **excellent localization**
- **competitive dense classification**
- **improved but still not dominant low-NEC sparse classification**
