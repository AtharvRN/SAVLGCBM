# ImageNet SAVLG Run Inventory (2026-04-29)

This note consolidates the completed ImageNet SAVLG dense and sparse runs found under `/workspace/savlg_imagenet_standalone_runs` on `atharv-rwx-pod`.

It is meant to answer two questions:

1. What do the completed `alpha=0.8` and `alpha=0.2` runs currently show?
2. Which sparse / GLM results are trustworthy enough to guide the next sweep?

## Scope

- Dense training runs considered:
  - `savlg_imagenet_full_7ep_a100_dense_20260428T115004Z_savlg-imagenet-full-a100-7ep-dense-xzm2t`
  - `savlg_imagenet_full_7ep_a100_dense_alpha02_20260429T124101Z_savlg-imagenet-full-a100-7ep-dense-alpha02-lpvjg`
- Sparse / GLM outputs considered:
  - earlier standalone GLM directories at workspace root
  - all `glm_*` directories inside the completed `alpha=0.2` run

## Dense Run Summary

| Run | Alpha | Status | Best concept val loss | Dense head val top-1 | Dense head val top-5 | Notes |
|---|---:|---|---:|---:|---:|---|
| `...dense-xzm2t` | 0.8 | complete | 0.2660 | 0.7467 | 0.8935 | main dense baseline |
| `...dense-alpha02-lpvjg` | 0.2 | complete | 0.3564 | 0.7522 | 0.8969 | concept branch converges worse, dense head still similar |
| `...dense-alpha00-xhgjn` | 0.0 | still running | n/a | n/a | n/a | only epoch 1-2 logs available |

### Dense interpretation

- `alpha=0.2` does **not** collapse dense classification.
- In fact, the dense head val top-1 is slightly higher for `alpha=0.2` than for `alpha=0.8`:
  - `0.7522` vs `0.7467`
- But the concept-training dynamics are worse for `alpha=0.2`:
  - best concept val loss is `0.3564` vs `0.2660`
  - final concept global val loss is much larger
- This means the dense classifier can still recover good class accuracy even when the concept branch is weaker. That matters because low-NEC behavior depends on concept quality, not just dense-head performance.

## Alpha=0.2 Dense Training Details

Completed run:

- Run dir:
  - `/workspace/savlg_imagenet_standalone_runs/savlg_imagenet_full_7ep_a100_dense_alpha02_20260429T124101Z_savlg-imagenet-full-a100-7ep-dense-alpha02-lpvjg`
- Key config:
  - `residual_alpha = 0.2`
  - `spatial_loss_mode = soft_align`
  - `loss_mask_w = 0.25`
  - `global_pos_weight = 1.0`
  - `concept_threshold = 0.15`
  - dense final layer, not GLM-SAGA final layer during training

Dense head result from `final_layer_summary.json`:

- val top-1: `0.752226`
- val top-5: `0.896915`
- best dense epoch: `1`

Concept training trend from `summary.json`:

- epoch 1 val loss: `0.9282`
- epoch 4 val loss: `0.5905`
- epoch 7 val loss: `0.3564`

This run clearly learned, but much more slowly and to a worse concept-loss endpoint than the `alpha=0.8` baseline.

## Earlier Alpha=0.2 Run

There is an earlier alpha `0.2` attempt:

- `savlg_imagenet_full_7ep_a100_dense_alpha02_20260429T100953Z_savlg-imagenet-full-a100-7ep-dense-alpha02-sctzh.bootstrap.log`

This one did **not** produce a completed run directory with summaries. The bootstrap log shows it was still recomputing precomputed targets when it was replaced by the corrected relaunch. It should be treated as a failed / superseded run, not a result-bearing experiment.

## NEC Evaluation Modes

There are three distinct evaluation modes in the current workspace, and mixing them leads to misleading comparisons.

### `exact_truncation`

- Train one sparse model.
- Then post-hoc truncate each class row to the top-`k` weights by `abs()`.
- NEC=5 means about `5 * 1000 = 5000` nonzeros total.
- This is the cleanest apples-to-apples NEC evaluation.

### `nearest_path_model`

- Train a path of sparse models at different lambdas.
- For each target NEC, choose the nearest path point.
- Different NEC buckets may map to the same sparse model if the path is coarse.

### `as_is_sparse_model`

- Train one sparse model and evaluate it directly.
- If the same model is reported for NEC=5,10,15,... then the NEC rows are not true exact-NEC numbers.

## Alpha=0.2 Sparse / GLM Inventory

All of the following live under:

- `/workspace/savlg_imagenet_standalone_runs/savlg_imagenet_full_7ep_a100_dense_alpha02_20260429T124101Z_savlg-imagenet-full-a100-7ep-dense-alpha02-lpvjg`

### Alpha=0.2 GLM runs

| Directory | `lam_max` | `n_iters` | `tol` | Val acc | NNZ | Weight sparsity | NEC eval mode | Notes |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `glm_one_lambda_lam1e3` | `0.0010` | 500 | `1e-6` | `0.5931` | 18,766 | `0.9956` | none recorded | very sparse, weak |
| `glm_one_lambda_lam7e4` | `0.0007` | 500 | `1e-4` | `0.7333` | 71,055 | `0.9835` | none recorded | much denser |
| `glm_one_lambda_lam5e4` | `0.0005` | 500 | `1e-4` | `0.7589` | 101,225 | `0.9765` | none recorded | best raw val acc among 500-iter runs, but very dense |
| `glm_one_lambda_lam5e4_tol1e6` | `0.0005` | 500 | `1e-6` | `0.7462` | 46,430 | `0.9892` | `exact_truncation` | most useful alpha=0.2 sparse result so far |
| `glm_one_lambda_lam5e4_tol1e7` | `0.0009` | 1000 | `1e-7` | `0.6524` | 23,469 | `0.9946` | none recorded | directory name is misleading; config says `lam_max=0.0009` |
| `glm_one_lambda_lam9e4_tol1e7` | n/a | n/a | n/a | n/a | n/a | n/a | failed / empty | empty directory |

### Alpha=0.2 Exact-NEC result

The only completed alpha `0.2` NEC test eval currently present is an `exact_truncation` eval:

- `glm_one_lambda_lam5e4_tol1e6/nec_eval_test.json`

Results:

| NEC | NNZ | Top-1 | Top-5 |
|---:|---:|---:|---:|
| 5 | 5,000 | 0.47636 | 0.73036 |
| 10 | 10,000 | 0.65124 | 0.85918 |
| 15 | 15,000 | 0.69702 | 0.89070 |
| 20 | 19,999 | 0.72080 | 0.90518 |
| 25 | 24,988 | 0.73522 | 0.91260 |
| 30 | 29,867 | 0.74322 | 0.91704 |

### Alpha=0.2 sparse interpretation

- `alpha=0.2` currently gives a **better NEC=5 result** than the earlier `alpha=0.8` exact-truncation run:
  - `0.4764` vs `0.4050`
- But this is still well below the paper target if the goal is “very strong low-NEC ImageNet”.
- Within alpha `0.2`, the main pattern is straightforward:
  - larger lambda gives the desired sparsity but hurts val acc
  - smaller lambda improves val acc but becomes too dense
- The `lam5e4_tol1e6` run is the best current compromise because it actually lands near the NEC targets and has test metrics.

## Earlier Alpha=0.8 Sparse Reference Points

These earlier GLM directories sit at workspace root and appear to come from the copied `alpha=0.8` artifact staged under `/tmp/savlg_imagenet_run` at the time. The exact source path is not preserved beyond that temp path, so provenance is slightly weaker than for the alpha `0.2` in-run directories.

### Alpha=0.8 reference sparse runs

| Directory | `lam_max` | `n_iters` | `max_glm_steps` | Val acc | NNZ | NEC eval mode |
|---|---:|---:|---:|---:|---:|---|
| `glm_path_lam1pe4_1000it_1step_gpu_cache` | `0.0001` | 1000 | 1 | `0.7933` | 285,291 | `exact_truncation` |
| `glm_path_lam7pe4_1000it_1step_gpu_cache` | `0.0007` | 1000 | 1 | `0.7063` | 46,132 | `as_is_sparse_model` |
| `glm_path_lam1p5e3_500it_100steps_gpu_cache` | `0.0010` | 800 | 100 | best path val `0.7452` | path varies | `nearest_path_model` |

### Alpha=0.8 Exact-truncation NEC reference

`glm_path_lam1pe4_1000it_1step_gpu_cache`:

| NEC | Top-1 | Top-5 |
|---:|---:|---:|
| 5 | 0.40498 | 0.64344 |
| 10 | 0.58900 | 0.81032 |
| 15 | 0.65876 | 0.86284 |
| 20 | 0.69326 | 0.88972 |
| 25 | 0.71388 | 0.90444 |
| 30 | 0.72896 | 0.91212 |

### Alpha=0.8 As-is sparse-model reference

`glm_path_lam7pe4_1000it_1step_gpu_cache`:

| NEC | Top-1 | Top-5 |
|---:|---:|---:|
| 5 | 0.71948 | 0.89630 |
| 10 | 0.71948 | 0.89630 |
| 15 | 0.71948 | 0.89630 |
| 20 | 0.71948 | 0.89630 |
| 25 | 0.71948 | 0.89630 |
| 30 | 0.71948 | 0.89630 |

This is not a true NEC sweep. The same untruncated sparse model was effectively reused for each NEC bucket.

### Alpha=0.8 Nearest-path-model reference

`glm_path_lam1p5e3_500it_100steps_gpu_cache`:

| NEC | Top-1 | Top-5 |
|---:|---:|---:|
| 5 | 0.54790 | 0.74284 |
| 10 | 0.54790 | 0.74284 |
| 15 | 0.54790 | 0.74284 |
| 20 | 0.54790 | 0.74284 |
| 25 | 0.56904 | 0.76310 |
| 30 | 0.62650 | 0.81454 |

This run is useful as a “path sweep can improve over the overly sparse 1-step model” signal, but it is not as clean as the exact-truncation NEC evals.

## Current takeaways

### 1. Dense alpha choice is not the bottleneck

- `alpha=0.2` dense head accuracy is already roughly on par with `alpha=0.8`.
- So the main problem is not dense classification quality.

### 2. Low-NEC sparse behavior is still the bottleneck

- Best documented `exact_truncation` runs:
  - `alpha=0.8`: NEC=5 top-1 `0.4050`
  - `alpha=0.2`: NEC=5 top-1 `0.4764`
- So alpha `0.2` is better than alpha `0.8` at NEC=5 in the current documented runs, but not yet strong enough.

### 3. The most useful next sweep should stay around the alpha=0.2 regime

Based on the completed alpha `0.2` GLM runs:

- `lam=1e-3` is too sparse and weak
- `lam=5e-4, tol=1e-6` is the best documented sparse operating point so far
- `lam=7e-4` and `lam=9e-4` degrade too much
- `lam=5e-4, tol=1e-4` is accurate but too dense to answer the low-NEC question

So the clean next sparse search should likely be:

- stay on the completed `alpha=0.2` dense artifact
- sweep around `lam_max` near `5e-4`

## Full ImageNet `mask_w=1.0` follow-up

Completed dense training run:

- `/workspace/savlg_imagenet_standalone_runs/savlg_imagenet_full_7ep_a100_dense_maskw1_20260430T092613Z_savlg-imagenet-full-a100-7ep-dense-maskw1-jsqmp`

Dense 50k val-tar eval from `final_layer_dense.pt`:

| Model | Top-1 | Top-5 | Notes |
|---|---:|---:|---|
| dense head | `0.75534` | `0.91732` | full 50k ImageNet val tar |

### `mask_w=1.0` sparse run now evaluated on 50k val

Sparse checkpoint:

- `/workspace/savlg_imagenet_standalone_runs/savlg_imagenet_full_7ep_a100_dense_maskw1_20260430T092613Z_savlg-imagenet-full-a100-7ep-dense-maskw1-jsqmp/glm_one_lambda_lam3e4_1000it_tol1e6_cpu_table_eval100`

Raw sparse model result:

| Lambda | Iterations | Tolerance | Val acc | NNZ | Weight sparsity | 50k val top-1 | 50k val top-5 | NEC eval mode |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `0.0003` | `1000` | `1e-6` | `0.7620` | `71,594` | `0.9834` | `0.76926` | `0.9312` | `as_is_sparse_model` |

Observed behavior during optimization:

| Iteration | Val acc | NNZ |
|---:|---:|---:|
| 100 | `0.7739` | `248,605` |
| 200 | `0.7708` | `157,056` |
| 300 | `0.7687` | `122,552` |
| 400 | `0.7670` | `104,549` |
| 1000 | `0.7620` | `71,594` |

Interpretation:

- `loss_mask_w=1.0` does not obviously fix the sparse frontier by itself.
- The run still stays far denser than NEC=5:
  - NEC=5 target is about `5,000` nonzeros
  - this raw sparse model still has `71,594`
- Even so, the raw sparse model preserves dense-level accuracy quite well on the full 50k val set:
  - dense head top-1 `0.75534`
  - raw sparse top-1 `0.76926`
- So this checkpoint remains useful for exact-truncation follow-up, but it is not naturally sparse enough at low NEC without an explicit truncation step.

### `mask_w=1.0` exact-truncation NEC eval on 50k val

To keep the saved truncated weights separate from the raw GLM run, the exact-truncation eval was staged under:

- `/workspace/savlg_imagenet_standalone_runs/savlg_imagenet_full_7ep_a100_dense_maskw1_20260430T092613Z_savlg-imagenet-full-a100-7ep-dense-maskw1-jsqmp/glm_one_lambda_lam3e4_1000it_tol1e6_cpu_table_eval100_exact_trunc`

This directory now contains:

- `final_layer_glm_saga.pt`
- `final_layer_normalization.pt`
- `W_g@NEC=5.pt`, `W_g@NEC=10.pt`, ..., `W_g@NEC=30.pt`
- `b_g@NEC=5.pt`, `b_g@NEC=10.pt`, ..., `b_g@NEC=30.pt`
- `nec_eval_val_tar.json`

Exact-truncation 50k val results:

| NEC | NNZ | Top-1 | Top-5 |
|---:|---:|---:|---:|
| 5 | 5,000 | `0.44848` | `0.69330` |
| 10 | 10,000 | `0.61948` | `0.83742` |
| 15 | 15,000 | `0.68408` | `0.88384` |
| 20 | 20,000 | `0.71458` | `0.90250` |
| 25 | 25,000 | `0.73392` | `0.91308` |
| 30 | 30,000 | `0.74518` | `0.91966` |

Key comparison for `mask_w=1.0`:

- raw sparse model at `71,594` nnz:
  - top-1 `0.76926`
  - top-5 `0.9312`
- exact NEC=5:
  - top-1 `0.44848`
  - top-5 `0.69330`

So `loss_mask_w=1.0` still does not close the low-NEC gap. It gives a viable dense-ish sparse model, but once the same weights are forced down to exact NEC targets, performance still drops sharply.
- use tighter tolerance and explicit `exact_truncation` NEC eval
- avoid one-off runs whose directory names do not match the actual config

## Caveats

- Dense run summaries currently give train/val metrics, not full ImageNet test-set dense eval, inside this workspace inventory.
- Earlier root-level GLM directories have weaker provenance because `source_run_dir.txt` only records a temp path such as `/tmp/savlg_imagenet_run`.
- `glm_one_lambda_lam5e4_tol1e7` is misnamed: its config actually uses `lam_max = 0.0009`.
- `glm_one_lambda_lam9e4_tol1e7` exists as an empty directory and should be treated as a failed / incomplete run.
