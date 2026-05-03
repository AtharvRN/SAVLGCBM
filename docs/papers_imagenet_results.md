# Paper ImageNet Results

This note summarizes the main completed ImageNet results for SAVLG, together with the comparison baselines used so far.

It is intended as the single paper-facing reference for:

- dense classification
- sparse / low-NEC classification
- localization
- main caveats on concept vocabularies and supervision

## Scope and Caveats

- All SAVLG numbers here come from completed runs in `/workspace/savlg_imagenet_standalone_runs`.
- VLG-CBM and SALF-CBM numbers come from pretrained checkpoints evaluated under the same ImageNet val-tar protocol.
- These comparisons are **checkpoint-native**:
  - VLG-CBM used its own concept set (`4300` concepts)
  - SALF-CBM used its own concept set (`4741` concepts)
  - SAVLG used the concept set bundled with each SAVLG artifact
- So this is **not** a perfectly vocabulary-matched comparison.
- Current localization evaluation is against the ImageNet val concept/box protocol used in our evaluator. It is useful, but it does **not** by itself prove localization against a completely independent human concept benchmark.

## Main Takeaways

1. Dense SAVLG ImageNet classification is already strong.
2. The new `b128` scratch-scale checkpoint substantially improves **low-NEC sparse performance**.
3. `alpha=0.2` improves low-NEC sparse accuracy over the earlier `alpha=0.8` reference.
4. `loss_mask_w=1.0` improves localization quality, but does **not** by itself solve the sparse frontier problem.
5. The best current SAVLG sparse localization result is clearly better than the current SALF-CBM and VLG-CBM localization baselines under this protocol.

## Dense Classification

### SAVLG dense full-ImageNet runs

| Run | Alpha | Dense val top-1 | Dense val top-5 | Notes |
|---|---:|---:|---:|---|
| SAVLG dense baseline | `0.8` | `0.7467` | `0.8935` | earlier main dense baseline |
| SAVLG dense | `0.2` | `0.7522` | `0.8969` | slightly better dense accuracy than alpha `0.8` |
| SAVLG dense, `mask_w=1.0` | `0.8`-family follow-up | `0.75534` | `0.91732` | full 50k val-tar eval from dense head |

Interpretation:

- Dense classification is **not** the main failure mode.
- `alpha=0.2` does not collapse dense accuracy.
- The sparse / NEC regime is the real problem.

## Sparse Classification

### Best current SAVLG sparse model

Best raw sparse SAVLG result evaluated on full 50k val:

| Model | Top-1 | Top-5 | NNZ | Notes |
|---|---:|---:|---:|---|
| SAVLG sparse, `mask_w=1.0`, `lambda=0.0003` | `0.76926` | `0.9312` | `71,594` | best current raw sparse SAVLG model on 50k val |

This is still much denser than low-NEC operation, so it is best viewed as a strong sparse classifier, not a true low-NEC result.

### Best current SAVLG exact-NEC result

Most useful exact-truncation SAVLG run so far:

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

Previous exact-truncation SAVLG run:

- artifact family: `alpha=0.2`
- GLM run: `glm_one_lambda_lam5e4_tol1e6`

| NEC | Top-1 | Top-5 |
|---:|---:|---:|
| 5 | `0.47636` | `0.73036` |
| 10 | `0.65124` | `0.85918` |
| 15 | `0.69702` | `0.89070` |
| 20 | `0.72080` | `0.90518` |
| 25 | `0.73522` | `0.91260` |
| 30 | `0.74322` | `0.91704` |

The `glm_sweep_lammax2e3` result is now the strongest current SAVLG low-NEC classification result.

### Additional completed SAVLG exact-NEC result

Completed 10ep scratch `mask_w=1.0` GLM artifact:

- run:
  - `savlg_imagenet_full_10ep_a100_alpha02_scratch_maskw1_20260502T103652Z_savlg-imagenet-full-a100-10ep-alpha02-scratch-maskw1-cnnm4`
- artifact:
  - `glm_one_lambda_lam2e4`
- persistent path:
  - `/workspace/savlg_imagenet_standalone_runs/.../glm_one_lambda_lam2e4`
- saved files:
  - `glm_path.pt`
  - `glm_path_metrics.json`
  - `nec_eval_test.json`
  - `W_g@NEC=*.pt`
  - `b_g@NEC=*.pt`

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
- Compared with `glm_sweep_lammax2e3`, this run is slightly lower on top-1 but higher on top-5.

### Earlier alpha `0.8` exact-NEC reference

| NEC | Top-1 | Top-5 |
|---:|---:|---:|
| 5 | `0.40498` | `0.64344` |
| 10 | `0.58900` | `0.81032` |
| 15 | `0.65876` | `0.86284` |
| 20 | `0.69326` | `0.88972` |
| 25 | `0.71388` | `0.90444` |
| 30 | `0.72896` | `0.91212` |

Interpretation:

- `alpha=0.2` is better than `alpha=0.8` at low NEC in the current runs.
- The new `glm_sweep_lammax2e3` result is now stronger than the current pretrained SALF-CBM NEC baseline in this workspace.

## Baseline Classification Reference

### SALF-CBM pretrained ImageNet checkpoint

| Model | Dense top-1 | Dense top-5 | ACC@NEC=5 | Avg ACC over NEC {5,10,15,20,25,30} |
|---|---:|---:|---:|---:|
| SAVLG best exact-NEC | n/a for this truncated-head eval | n/a for this truncated-head eval | `0.76304` | `0.78578` |
| SALF-CBM pretrained | `0.75564` | `0.91632` | `0.53534` | `0.69784` |

### VLG-CBM

For localization we evaluated a staged pretrained VLG-CBM checkpoint copy. For classification, the main current VLG-style reference in the docs is localization-oriented rather than a fully consolidated dense-vs-NEC table in this note.

## Localization

### Headline localization metrics

Current paper-friendly localization metrics:

- `LocAcc@0.3`: localization accuracy at IoU `0.3`
- `LocAcc@0.5`: localization accuracy at IoU `0.5`
- `MeanIoU`: mask IoU using the heatmap mean as the threshold

### Dense localization comparison

| Model | LocAcc@0.3 | LocAcc@0.5 | MeanIoU |
|---|---:|---:|---:|
| VLG-CBM Grad-CAM | `0.5246` | `0.3572` | `0.2405` |
| SALF-CBM native | `0.5985` | `0.4375` | `0.2691` |
| SAVLG native | `0.6542` | `0.4873` | `0.3449` |

Interpretation:

- Dense SAVLG already outperforms both VLG-CBM Grad-CAM and SALF-CBM on the main localization metrics.

### Best current sparse SAVLG localization result

This uses the best saved sparse SAVLG model:

- artifact:
  - `.../glm_one_lambda_lam3e4_1000it_tol1e6_cpu_table_eval100`

Headline metrics:

| Model | LocAcc@0.3 | LocAcc@0.5 | MeanIoU |
|---|---:|---:|---:|
| SAVLG sparse best current | `0.90894` | `0.60047` | `0.42558` |

Other localization metrics for that sparse model:

| Metric | Value |
|---|---:|
| `soft_iou` | `0.40754` |
| `mass_in_gt` | `0.59783` |
| `point_hit` | `0.75024` |
| `mask_iou@0.5` | `0.45055` |

Interpretation:

- This is the strongest SAVLG localization result recorded so far.
- It is clearly stronger than the current SALF-CBM localization baseline under the same evaluation protocol.

## Overall Ranking So Far

### Classification

- Best dense SAVLG top-1 so far: `0.75534`
- Best raw sparse SAVLG top-1 so far: `0.76926`
- Best SAVLG exact `NEC=5` top-1 so far: `0.76304`
- SALF-CBM pretrained `NEC=5` top-1: `0.53534`

So:

- SAVLG is competitive in dense classification.
- SAVLG is strong in high-NNZ sparse classification.
- The new SAVLG low-NEC result is stronger than the current pretrained SALF-CBM reference in this workspace.

### Localization

- Best current localization: **SAVLG sparse**
- Next: **SAVLG dense**
- Then: **SALF-CBM**
- Then: **VLG-CBM Grad-CAM**

So localization is currently the strongest part of the SAVLG ImageNet story.

## Recommended Paper Framing

The current evidence supports the following claims:

1. SAVLG yields strong dense ImageNet classification.
2. SAVLG yields strong native localization maps, beating VLG-CBM Grad-CAM and SALF-CBM on the current ImageNet localization protocol.
3. SAVLG sparse models can preserve strong accuracy at moderate sparsity.
4. The earlier exact low-NEC gap has been substantially reduced by the new `glm_sweep_lammax2e3` run.

The current evidence does **not** yet fully support:

1. “SAVLG is best at low-NEC ImageNet classification across all possible vocabulary-matched baselines.”
2. “Localization is proven against a completely independent human concept benchmark.”

Those remain the main gaps for the next round of experiments.
