# CUB LF-CBM, SALF-CBM, SAVLG-CBM, and VLG-CBM Results

This document records the completed `CUB` runs and the exact pod artifact paths currently in use for the reproduction effort.

## Pod workspaces

- Shared workspace root: `/workspace/SAVLGCBM`
- `LF-CBM` run and NEC sweep were completed on: `atharv-rwx-pod`
- `SALF-CBM` run was completed on: `atharv-rwx-pod`
- `VLG-CBM` training and NEC sweep were completed on: `atharv-rwx-pod-2`
- Cross-backbone follow-up runs and reduced-scope NEC sweeps were completed on: `atharv-rwx-pod-2`

## Local pulled logs

These pod logs were copied into the local repo for inspection:

- `logs/cub_lf_resnet18_cub_nec_fast.log`
- `logs/cub_vlg_clip_rn50_nec_fast.log`
- `logs/cub_salf_resnet18_cub_nec_fast.log`
- `logs/cub_salf_resnet18_cub_pod2_small.log`

Additional pod-only logs used for the SAVLG first-pass run:

- `/workspace/SAVLGCBM/logs/cub_savlg_tmp.log`
- `/workspace/SAVLGCBM/logs/cub_savlg_nec_fast_rerun.log`

## CUB summary tables

### By model and backbone

| Model | Backbone | Full test acc | ACC@5 | AVGACC | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `LF-CBM` | `clip_RN50` | `0.6634` | `0.3402` | `0.5612` | Initial unified-bank run |
| `LF-CBM` | `resnet18_cub` | `0.7402` | `0.5147` | `0.6817` | Original LF bank + fixed LF similarity + tuned NEC |
| `SALF-CBM` | `clip_RN50` | `0.3573` | `0.1233` | `0.2429` | Underperforming run, likely protocol mismatch |
| `SALF-CBM` | `resnet18_cub` | `0.7320` | `0.5335` | `0.6826` | Original LF bank + fixed LF similarity + tuned NEC |
| `SAVLG-CBM` | `resnet18_cub` | `0.7459` | `0.6917` | `0.7170` | First-pass VLG-annotation adaptation, fast NEC rerun |
| `VLG-CBM` | `resnet18_cub` | `0.7594` | `0.7546` | `0.7556` | Upstream-style `CUB` setup |
| `VLG-CBM` | `clip_RN50` | `0.6553` | `0.6247` | `0.6504` | Cross-backbone follow-up, fast NEC rerun |

### By backbone

| Backbone | LF-CBM full / AVGACC | SALF-CBM full / AVGACC | SAVLG-CBM full / AVGACC | VLG-CBM full / AVGACC |
| --- | --- | --- | --- |
| `clip_RN50` | `0.6634 / 0.5612` | `0.3573 / 0.2429` | `N/A` | `0.6553 / 0.6504` |
| `resnet18_cub` | `0.7402 / 0.6817` | `0.7320 / 0.6826` | `0.7459 / 0.7170` | `0.7594 / 0.7556` |

Interpretation:
- `resnet18_cub` is clearly the healthier `CUB` backbone across all three methods in this codebase.
- After fixing the LF port's similarity function and switching back to the original LF concept bank, `LF-CBM resnet18_cub` now reaches `0.7402` dense accuracy and matches the untouched original LF repo run on the same pod.
- After restoring the LF-style similarity and rerunning NEC, `SALF-CBM resnet18_cub` reaches `0.7320` dense accuracy and `0.6826` `AVGACC`, essentially matching the corrected `LF-CBM` sparse performance.
- The `SALF-CBM clip_RN50` result should not be treated as the representative `SALF` result on `CUB`.
- First-pass `SAVLG-CBM` on `resnet18_cub` is clearly stronger than `LF-CBM` and `SALF-CBM` on the same backbone, but it is still below `VLG-CBM`.
- Relative to `VLG-CBM` on `resnet18_cub`, `SAVLG-CBM` is lower by `0.0135` in dense test accuracy and `0.0386` in `AVGACC`.

### SAVLG follow-up sprint note

- Completed soft-box follow-up run:
  - dense test accuracy `0.7442`
  - `ACC@5 = 0.7026`
  - `AVGACC = 0.7134`
- This is a small regression from the original `SAVLG-CBM resnet18_cub` baseline (`0.7459` dense, `0.7170` `AVGACC`), so soft box occupancy targets are not the new default.
- Completed dense `VLG` distillation follow-up:
  - dense test accuracy `0.7444`
  - `ACC@5 = 0.7069`
  - `AVGACC = 0.7193`
- Distillation is the strongest sparse `SAVLG-CBM` follow-up so far, improving `AVGACC` over the current `SAVLG` baseline (`0.7193` vs `0.7170`), but it still does not become the new default because dense accuracy remains slightly below the dense baseline `0.7459`.
- Completed WILDCAT local top-k follow-up run:
  - dense test accuracy `0.7380`
  - NEC not run because the dense result was a clear regression versus the current `SAVLG` baseline and the completed soft-box follow-up
- This local top-k auxiliary branch should not replace the current `lse` local MIL setting.
- Completed OICR-style self-refinement follow-up:
  - dense test accuracy `0.7380`
  - NEC not run because the dense result was a clear regression versus the current `SAVLG` baseline and the stronger soft-box / distillation follow-ups
- Completed selective localization weighting follow-up:
  - dense test accuracy `0.7370` (`0.7369602763385147`)
  - NEC not promoted; despite passing `>0.73`, dense regressed materially versus the `SAVLG` baseline `0.7459`
- Completed MIL distillation ablation sprint (`w010/w025/w035`) and NEC-path follow-up:
  - best dense result plateaued around `0.7484`
  - `w010` dense complete: `0.7483592400690846`
    - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_03_20_56_45`
  - `w010` tuned NEC complete:
    - `ACC@5=0.5059`, `ACC@10=0.6955`, `ACC@15=0.7288`, `ACC@20=0.7297`, `ACC@25=0.7297`, `ACC@30=0.7297`, `AVGACC=0.6866`
  - `w025` dense complete: `0.7481865284974093`
    - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_03_22_57_19`
  - `w025` tuned NEC complete:
    - `ACC@5=0.5048`, `ACC@10=0.6953`, `ACC@15=0.7287`, `ACC@20=0.7294`, `ACC@25=0.7294`, `ACC@30=0.7294`, `AVGACC=0.6862`
  - `w035` dense complete: `0.7484`
    - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_03_23_06_30`
  - `w035` tuned NEC complete:
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_mil_distill_w035_nec_tuned.log`
    - `ACC@5=0.4770`, `ACC@10=0.6946`, `ACC@15=0.7288`, `ACC@20=0.7295`, `ACC@25=0.7295`, `ACC@30=0.7295`, `AVGACC=0.6815`
  - dense-control run for NEC-path scheduling also reached `0.7483592400690846`
    - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_03_23_47_35`
  - completed NEC-path scheduling variant:
    - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_03_23_47_35_necpath_lam0005_s30`
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_necpath_dense_control_nec_lam0005_s30.log`
    - `ACC@5=0.7033`, `ACC@10=0.7168`, `ACC@15=0.7168`, `ACC@20=0.7168`, `ACC@25=0.7168`, `ACC@30=0.7168`, `AVGACC=0.7145`
  - interpretation:
    - `w035` confirms that pushing the distillation weight higher hurts sparse recovery instead of helping it
    - the NEC-path schedule is the strongest result recovered from this distill-line family, but it still remains below the later clean-global `SAVLG` checkpoint, so it is not the new default
- Completed clean global-concept rerun with binary-threshold targets and `adam` for `75` epochs:
  - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_03_35_49`
  - dense train log: `/workspace/SAVLGCBM/logs/cub_savlg_clean_globalconcept_binary_adam_e75.log`
  - NEC log: `/workspace/SAVLGCBM/logs/cub_savlg_clean_globalconcept_binary_adam_e75_nec150.log`
  - dense test accuracy `0.7492`
  - `ACC@5 = 0.7440`
  - `ACC@10 = 0.7478`
  - `ACC@15 = 0.7487`
  - `ACC@20 = 0.7482`
  - `ACC@25 = 0.7492`
  - `ACC@30 = 0.7494`
  - `AVGACC = 0.7479`
- This clean global-concept run is the current best overall `SAVLG-CBM` result in this repo and the fairest comparison point against the corrected `SALF-CBM` baseline.
- Completed soft-containment dense follow-up on top of that clean-global baseline:
  - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13`
  - dense test accuracy `0.7451` (`0.7450777202072539`)
  - dense gate decision:
    - clears the `0.74` threshold, so NEC was promoted
    - does not beat the current clean-global dense best `0.7492`
  - completed tuned NEC follow-up:
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_softcontain_v1_nec150.log`
    - `ACC@5 = 0.7413`
    - `ACC@10 = 0.7444`
    - `ACC@15 = 0.7447`
    - `ACC@20 = 0.7442`
    - `ACC@25 = 0.7447`
    - `ACC@30 = 0.7449`
    - `AVGACC = 0.7440`
  - sparse interpretation:
    - this remains slightly below the clean-global `SAVLG` checkpoint on `ACC@5` (`0.7440`) and `AVGACC` (`0.7479`)
    - soft containment has therefore not yet become the new default on classification/sparsity metrics alone
  - completed native-map localization follow-up:
    - output: `/workspace/SAVLGCBM/results/native_savlg_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`
    - mean IoU `0.0799`
    - `mAP@0.3 = 0.9500`
    - `mAP@0.5 = 0.9682`
    - `mAP@0.7 = 0.9734`
    - point hit rate `0.8518`
    - point coverage `0.0500`
  - completed Grad-CAM localization follow-up:
    - output: `/workspace/SAVLGCBM/results/gradcam_savlg_softcontain_v1_full.json`
    - mean IoU `0.0862`
    - `mAP@0.3 = 0.0332`
    - `mAP@0.5 = 0.0127`
    - `mAP@0.7 = 0.00642`
    - point hit rate `0.7765`
    - point coverage `0.00650`
  - localization interpretation:
    - soft containment clearly improved bbox localization over the clean-global `SAVLG` checkpoint, especially for native maps and modestly for Grad-CAM
    - paper-style point localization remains the main failure mode:
      - Grad-CAM hit rate improved over clean-global `SAVLG` (`0.7765` vs `0.7374`)
      - Grad-CAM coverage fell further (`0.00650` vs `0.0114`)
      - corrected `SALF` still remains far stronger on point hit / coverage (`0.9427 / 0.9513`)
    - because classification/sparsity still regressed versus the clean-global best and localization still needs cleaner point evidence, the next ranked sprint moves to a dual-branch global-plus-spatial architecture
- Dual-branch global-plus-spatial follow-up on top of the clean-global soft-containment setup:
  - dense checkpoint:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12`
  - dense test accuracy:
    - `0.744559585492228`
  - localization outputs:
    - native:
      - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`
      - mean IoU `0.10386529624037594`
      - `mAP@0.3 = 0.22793726036719067`
      - `mAP@0.5 = 0.22849189875236445`
      - `mAP@0.7 = 0.22686268117877612`
      - point hit rate `0.9317171436510142`
      - point coverage `0.41429126437112396`
    - Grad-CAM:
      - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_softcontain_v1_full.json`
      - mean IoU `0.07861747122510306`
      - `mAP@0.3 = 0.015886130176933487`
      - `mAP@0.5 = 0.007540785230536763`
      - `mAP@0.7 = 0.005351927302307943`
      - point hit rate `0.8132530120481928`
      - point coverage `0.0023885923133372664`
  - sparse recovery status:
    - base `nec150` collapsed:
      - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12/metrics.csv`
      - `NEC = 655.445014834404`
      - `Accuracy = 0.01295336801558733`
    - `nec400` collapsed:
      - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s400/metrics.csv`
      - `NEC = 645.5450026392937`
      - `Accuracy = 0.0321243517100811`
    - `nec800` collapsed:
      - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s800/metrics.csv`
      - `NEC = 655.3799834251404`
      - `Accuracy = 0.024352332577109337`
    - `nec1600` collapsed:
      - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s1600/metrics.csv`
      - `NEC = 647.7299939990044`
      - `Accuracy = 0.029879102483391762`
    - `nec3200` collapsed:
      - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s3200/metrics.csv`
      - `NEC = 649.5349955558777`
      - `Accuracy = 0.026597581803798676`
  - interpretation:
    - dual-branch materially improves native-map localization over the earlier clean-global and single-branch soft-containment checkpoints
    - native point-localization hit / coverage (`0.9317 / 0.4143`) is the strongest `SAVLG` localization evidence recovered so far in this repo
    - Grad-CAM point hit rate also improves over prior `SAVLG`, but Grad-CAM coverage remains extremely weak
    - dense accuracy still trails the clean-global best (`0.7446` vs `0.7492`)
    - sparse behavior remained unusable even after `400`, `800`, `1600`, and `3200` step recovery reruns, so this sprint is a localization-positive but classification/sparsity-negative result and does not become the new default
- Completed dual-branch local-MIL follow-up on top of the dual-branch soft-containment setup:
  - dense checkpoint:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_21_27_29`
  - dense log:
    - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_localmil_lse_v1.log`
  - dense test accuracy:
    - `0.7500863557858376`
  - dense gate decision:
    - clears the `0.74` NEC threshold
    - slightly exceeds the prior clean-global dense best `0.7492`
  - completed NEC follow-up:
    - log:
      - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_localmil_lse_v1_nec150.log`
    - metrics file:
      - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_21_27_29/metrics.csv`
    - `ACC@5 = 0.7187`
    - `ACC@10 = 0.7249`
    - `ACC@15 = 0.7271`
    - `ACC@20 = 0.7288`
    - `ACC@25 = 0.7301`
    - `ACC@30 = 0.7314`
    - `AVGACC = 0.7268`
  - completed native-map localization:
    - output:
      - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_localmil_lse_v1_full_meanthr_bs32_nw8_nocache.json`
    - mean IoU `0.09073396545708716`
    - `mAP@0.3 = 0.9134870013023054`
    - `mAP@0.5 = 0.9319806619771949`
    - `mAP@0.7 = 0.9366440387587317`
    - point hit rate `0.9529931699477702`
    - point coverage `0.035814495589737684`
  - completed Grad-CAM localization:
    - output:
      - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_localmil_lse_v1_full.json`
    - mean IoU `0.07988204531850018`
    - `mAP@0.3 = 0.0266030624693556`
    - `mAP@0.5 = 0.010985647975612694`
    - `mAP@0.7 = 0.006500362319831989`
    - point hit rate `0.729794933655006`
    - point coverage `0.011928572456365022`
  - interpretation:
    - this is now the best dense-only `SAVLG` checkpoint in the repo
    - the dense gain did not translate into competitive sparse behavior, so the current best complete `SAVLG` checkpoint remains the clean-global run at dense `0.7492`, `ACC@5 = 0.7440`, and `AVGACC = 0.7479`
    - native-map localization is mixed:
      - native point hit rises to `0.9530`, but coverage stays very low at `0.0358`
    - native mean IoU still trails the stronger dual-branch soft-containment checkpoint (`0.0907` vs `0.1039`)
    - Grad-CAM localization remains roughly flat relative to earlier `SAVLG` checkpoints and still fails badly on paper-style point coverage
    - this sprint is therefore dense-positive but sparse-negative and not a clear localization win, so it does not become the new default
- Completed dual-branch local-MIL plus `mlp(h=1)` controlled rerun:
  - dense checkpoint:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_11_21`
  - dense log:
    - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_localmil_lse_mlp_h1_v1.log`
  - dense test accuracy:
    - `0.7347150259067358`
  - dense gate decision:
    - fails the `0.74` threshold in `TASK.md`
    - NEC was not promoted
    - localization follow-up was not promoted
  - interpretation:
    - swapping the dual-branch local-MIL recipe from a linear concept head to `mlp(h=1)` is a clear dense regression relative to the matched linear-head checkpoint (`0.7347` vs `0.7501`)
    - this rerun also falls below the clean-global `SAVLG` baseline (`0.7492`), so it should be treated as a negative result and deprioritized
- Multiscale spatial-branch dense and sparse stages are now complete; localization evals remain in flight:
  - dense checkpoint:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_26_28`
  - dense log:
    - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_v1.log`
  - dense test accuracy:
    - `0.7506044905008635`
  - dense gate decision:
    - clears the `0.74` NEC threshold in `TASK.md`
    - is now the strongest dense-only `SAVLG` checkpoint in the repo (`0.7506` vs `0.7501`)
    - also exceeds the best complete clean-global `SAVLG` dense checkpoint (`0.7492`)
  - follow-up jobs launched from this checkpoint:
    - NEC log:
      - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_v1_nec150.log`
    - native localization output:
      - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full_meanthr_bs32_nw8_nocache.json`
    - Grad-CAM localization output:
      - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full.json`
  - finalized sparse artifacts:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_26_28/metrics.csv`
    - exact saved heads:
      - `W_g@NEC=5.pt`
      - `W_g@NEC=10.pt`
      - `W_g@NEC=15.pt`
      - `W_g@NEC=20.pt`
      - `W_g@NEC=25.pt`
      - `W_g@NEC=30.pt`
  - exact sparse metrics from the saved `W_g@NEC=*` heads:
    - `ACC@5 = 0.7445595860481262`
    - `ACC@10 = 0.7481865286827087`
    - `ACC@15 = 0.7490500807762146`
    - `ACC@20 = 0.7485319375991821`
    - `ACC@25 = 0.7487046718597412`
    - `ACC@30 = 0.7488774061203003`
    - `AVGACC = 0.7479850351810455`
  - sparse-eval loader fix required for this checkpoint:
    - [evaluations/sparse_utils.py](/Users/atharvramesh/Projects/CBM/SAVLGCBM/evaluations/sparse_utils.py) now rebuilds SAVLG checkpoints with `build_savlg_concept_layer(args, backbone, len(concepts))`
    - reason:
      - the generic spatial-layer loader dropped the multiscale `conv4_proj` and `conv5_proj` weights during NEC reload
  - current status:
    - this is now the strongest fully measured `SAVLG` classification checkpoint in the repo so far:
      - dense `0.7506`
      - `ACC@5 0.7446`
      - `AVGACC 0.7480`
    - it slightly exceeds the prior best complete clean-global `SAVLG` checkpoint on both dense and sparse metrics, but still trails the current `VLG-CBM` frontier
    - both localization evals are still in flight, so this checkpoint is not yet a complete promoted localization-aware result
- Completed localization follow-up artifacts on the live pod:
  - completed native-map bbox comparison from `/workspace/SAVLGCBM/results/native_map_compare_savlg_vs_salf_full_meanthr_bs32_nw8_nocache_20260404T184114.json`
    - `SAVLG`: mean IoU `0.0233`, `mAP@0.3 = 0.00620`, `mAP@0.5 = 0.00505`, `mAP@0.7 = 0.00458`
    - `SALF`: mean IoU `0.0778`, `mAP@0.3 = 0.00645`, `mAP@0.5 = 0.00500`, `mAP@0.7 = 0.00456`
  - completed Grad-CAM bbox comparison from:
    - `/workspace/SAVLGCBM/results/gradcam_savlg_best_full.json`
    - `/workspace/SAVLGCBM/results/gradcam_salf_best_full.json`
    - `SAVLG`: mean IoU `0.0797`, `mAP@0.3 = 0.0266`, `mAP@0.5 = 0.0109`, `mAP@0.7 = 0.00648`
    - `SALF`: mean IoU `0.0789`, `mAP@0.3 = 0.00661`, `mAP@0.5 = 0.00512`, `mAP@0.7 = 0.00456`
  - completed Grad-CAM point-localization eval for corrected `SALF` from:
    - `/workspace/SAVLGCBM/results/gradcam_point_salf_best_full.json`
    - hit rate `0.9427`
    - coverage `0.9513`
    - matched part hits / total `62328 / 66114`
  - completed Grad-CAM point-localization eval for clean-global `SAVLG` from:
    - `/workspace/SAVLGCBM/results/gradcam_point_savlg_best_full.json`
    - hit rate `0.7374`
    - coverage `0.0114`
    - matched part hits / total `584 / 792`
  - interpretation:
    - `SAVLG` is not yet better than `SALF` on native-map mean IoU
    - `SAVLG` is clearly stronger than `SALF` on Grad-CAM bbox localization mAP
    - the point-localization comparison reverses that story sharply:
      - current clean-global `SAVLG` Grad-CAM maps are too sparse or off-target for the CUB point metric
      - corrected `SALF` remains much better on paper-style point hit / coverage
    - this completed localization gate is the reason the next dense-first sprint moved to a soft containment loss instead of another hard-mask or NEC follow-up
- Main takeaway from the stopped loop:
  - teacher distillation remains the only model-change direction that consistently helped
  - later MIL distillation ablations improved dense accuracy over the `SAVLG` baseline (`~0.7484` vs `0.7459`), but tuned NEC regressed versus the earlier sparse-best distillation run
  - the best complete `SAVLG-CBM` result is still the clean global-concept checkpoint (`dense 0.7492`, `ACC@5 0.7440`, `AVGACC 0.7479`)
  - the best dense-only `SAVLG-CBM` checkpoint is now the in-flight multiscale spatial-branch run (`dense 0.7506044905008635`)
  - the current bottleneck is sparse evaluation / NEC path quality rather than dense training
- NEC interpretation note:
  - `metrics.csv` stores the full GLM sparsity path, not the final `ACC@5` / `AVGACC` summary table
  - low-NEC early rows in `metrics.csv` should not be read directly as the reported `ACC@5` metric

## LF-CBM

Run description:
- Model: `LF-CBM`
- Dataset: `CUB`
- Concept bank: `concept_files/cub_filtered.txt` from upstream `VLG-CBM`
- Backbone: `clip_RN50`
- CLIP pseudo-label model: `ViT-B/16`
- Protocol: matched official train/test split semantics via `lf_original_protocol=true`

Checkpoint and artifact directory:
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39`

Primary checkpoint files:
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/W_c.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/W_g.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/b_g.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/proj_mean.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/proj_std.pt`

Run metadata and logs:
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/args.txt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/artifacts.json`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/method_log.json`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/train.log`

Full-accuracy result files:
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/train_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/val_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/test_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/metrics.txt`

Sparse / NEC result files:
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/metrics.csv`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/W_g@NEC=5.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/W_g@NEC=10.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/W_g@NEC=15.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/W_g@NEC=20.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/W_g@NEC=25.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/W_g@NEC=30.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/b_g@NEC=5.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/b_g@NEC=10.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/b_g@NEC=15.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/b_g@NEC=20.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/b_g@NEC=25.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_19_56_39/b_g@NEC=30.pt`

Recorded metrics:
- Train accuracy: `0.8626`
- Val accuracy: `0.6634`
- Test accuracy: `0.6634`
- `ACC@5`: `0.3402`
- `ACC@10`: `0.5161`
- `ACC@15`: `0.5900`
- `ACC@20`: `0.6290`
- `ACC@25`: `0.6402`
- `ACC@30`: `0.6516`
- `AVGACC`: `0.5612`

## SALF-CBM

Run description:
- Model: `SALF-CBM`
- Dataset: `CUB`
- Concept bank: `concept_files/cub_filtered.txt` from upstream `VLG-CBM`
- Backbone: `clip_RN50`
- CLIP pseudo-label model: `ViT-B/16`
- Spatial target source: `prompt_grid`
- Grid: `7 x 7`
- Prompt radius: `32`

Checkpoint and artifact directory:
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29`

Primary checkpoint files:
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/concept_layer.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/W_g.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/b_g.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/proj_mean.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/proj_std.pt`

Run metadata and logs:
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/args.txt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/artifacts.json`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/method_log.json`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/train.log`

Full-accuracy result files:
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/train_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/val_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/test_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/metrics.txt`

Recorded metrics:
- Train accuracy: `0.8947`
- Val accuracy: `0.3573`
- Test accuracy: `0.3573`

Sparse / NEC result files:
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/metrics.csv`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/W_g@NEC=5.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/W_g@NEC=10.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/W_g@NEC=15.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/W_g@NEC=20.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/W_g@NEC=25.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/W_g@NEC=30.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/b_g@NEC=5.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/b_g@NEC=10.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/b_g@NEC=15.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/b_g@NEC=20.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/b_g@NEC=25.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_01_21_05_29/b_g@NEC=30.pt`

Recorded NEC metrics:
- `ACC@5`: `0.1233`
- `ACC@10`: `0.2003`
- `ACC@15`: `0.2478`
- `ACC@20`: `0.2784`
- `ACC@25`: `0.2976`
- `ACC@30`: `0.3098`
- `AVGACC`: `0.2429`

Current interpretation:
- This `SALF-CBM` run is substantially worse than the matched `LF-CBM` run on the same `CUB` concept bank and `clip_RN50` backbone.
- The gap is large both in dense accuracy and sparse / NEC accuracy.
- For the current reproduction effort, this should be treated as a likely implementation or protocol mismatch, not as expected evidence that SALF is genuinely worse than LF on `CUB`.

## VLG-CBM

Run description:
- Model: `VLG-CBM`
- Dataset: `CUB`
- Concept bank: `concept_files/cub_filtered.txt` from upstream `VLG-CBM`
- Backbone: `resnet18_cub`

Checkpoint and artifact directory:
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42`

Primary checkpoint files:
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/cbl.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/final.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/train_concept_features_mean.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/train_concept_features_std.pt`

Run metadata and logs:
- `/workspace/SAVLGCBM/logs/cub_vlg_pod2.log`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/args.txt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/artifacts.json`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/metrics.txt`

Recorded full-accuracy metrics:
- Test accuracy: `0.7594`

Sparse / NEC result files:
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/metrics.csv`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/W_g@NEC=5.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/W_g@NEC=10.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/W_g@NEC=15.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/W_g@NEC=20.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/W_g@NEC=25.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/W_g@NEC=30.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/b_g@NEC=5.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/b_g@NEC=10.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/b_g@NEC=15.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/b_g@NEC=20.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/b_g@NEC=25.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_20_50_42/b_g@NEC=30.pt`

Recorded NEC metrics:
- `ACC@5`: `0.7546`
- `ACC@10`: `0.7554`
- `ACC@15`: `0.7556`
- `ACC@20`: `0.7558`
- `ACC@25`: `0.7560`
- `ACC@30`: `0.7561`
- `AVGACC`: `0.7556`

## Cross-backbone follow-up runs

These runs were added to compare the methods under swapped backbones:

- `LF-CBM` with `resnet18_cub`
- `SALF-CBM` with `resnet18_cub`
- `VLG-CBM` with `clip_RN50`

The sparse evals below used the reduced-scope fast rerun path with `--n_iters 1000`.

### LF-CBM with `resnet18_cub`

Run description:
- Model: `LF-CBM`
- Dataset: `CUB`
- Concept bank: `concept_files/cub_filtered.txt` from upstream `VLG-CBM`
- Backbone: `resnet18_cub`
- CLIP pseudo-label model: `ViT-B/16`

Checkpoint and artifact directory:
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_22_22_38`

Run metadata and logs:
- Pod train log: `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_22_22_38/train.log`
- Pod NEC log: `/workspace/SAVLGCBM/logs/cub_lf_resnet18_cub_nec_fast.log`
- Local NEC log copy: `logs/cub_lf_resnet18_cub_nec_fast.log`

Full-accuracy result files:
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_22_22_38/train_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_22_22_38/val_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_22_22_38/test_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_22_22_38/metrics.txt`

Sparse / NEC result files:
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_22_22_38/metrics.csv`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_22_22_38/W_g@NEC=5.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_22_22_38/W_g@NEC=10.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_22_22_38/W_g@NEC=15.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_22_22_38/W_g@NEC=20.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_22_22_38/W_g@NEC=25.pt`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_01_22_22_38/W_g@NEC=30.pt`

Recorded metrics:
- Train accuracy: `0.9985`
- Val accuracy: `0.6972`
- Test accuracy: `0.6972`
- `ACC@5`: `0.4532`
- `ACC@10`: `0.6345`
- `ACC@15`: `0.6763`
- `ACC@20`: `0.6891`
- `ACC@25`: `0.6931`
- `ACC@30`: `0.6965`
- `AVGACC`: `0.6405`

### LF-CBM with `resnet18_cub`, original LF bank, and fixed LF similarity

Run description:
- Model: `LF-CBM`
- Dataset: `CUB`
- Concept bank: `concept_files/label_free_cbm_cub_filtered.txt`
- Backbone: `resnet18_cub`
- CLIP pseudo-label model: `ViT-B/16`
- Protocol: `lf_original_protocol=true`
- Key fix: exact upstream LF cubed-cosine similarity restored in `methods/lf.py`

Checkpoint and artifact directory:
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_03_04_22_18`

Run metadata and logs:
- Pod train log: `/workspace/SAVLGCBM/logs/cub_lf_resnet18_cub_lfbank_original_exact_fixedsim.log`
- Pod tuned NEC log: `/workspace/SAVLGCBM/logs/cub_lf_resnet18_cub_lfbank_original_exact_fixedsim_nec_tuned2.log`

Full-accuracy result files:
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_03_04_22_18/train_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_03_04_22_18/val_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/lf_cbm_cub_2026_04_03_04_22_18/test_metrics.json`

Recorded dense metrics:
- Train accuracy: `1.0000`
- Val accuracy: `0.7402`
- Test accuracy: `0.7402`

Recorded tuned NEC metrics:
- Sweep settings: `--lam 0.01 --max_glm_steps 40`
- `ACC@5`: `0.5147`
- `ACC@10`: `0.6808`
- `ACC@15`: `0.7155`
- `ACC@20`: `0.7261`
- `ACC@25`: `0.7266`
- `ACC@30`: `0.7266`
- `AVGACC`: `0.6817`

Current interpretation:
- This is the first LF run in this repo that matches the untouched original `Label-free-CBM` repo on the same pod and backbone.
- The main missing piece in the port was the LF similarity function and how it was applied during projection training and interpretability filtering.

### VLG-CBM with `clip_RN50`

Run description:
- Model: `VLG-CBM`
- Dataset: `CUB`
- Concept bank: `concept_files/cub_filtered.txt` from upstream `VLG-CBM`
- Backbone: `clip_RN50`

Checkpoint and artifact directory:
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_22_22_37`

Run metadata and logs:
- Pod train log: `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_22_22_37/train.log`
- Pod NEC log: `/workspace/SAVLGCBM/logs/cub_vlg_clip_rn50_nec_fast.log`
- Local NEC log copy: `logs/cub_vlg_clip_rn50_nec_fast.log`

Full-accuracy result files:
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_22_22_37/metrics.txt`

Sparse / NEC result files:
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_22_22_37/metrics.csv`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_22_22_37/W_g@NEC=5.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_22_22_37/W_g@NEC=10.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_22_22_37/W_g@NEC=15.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_22_22_37/W_g@NEC=20.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_22_22_37/W_g@NEC=25.pt`
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_01_22_22_37/W_g@NEC=30.pt`

Recorded metrics:
- Test accuracy: `0.6553`
- `ACC@5`: `0.6247`
- `ACC@10`: `0.6470`
- `ACC@15`: `0.6547`
- `ACC@20`: `0.6594`
- `ACC@25`: `0.6596`
- `ACC@30`: `0.6570`
- `AVGACC`: `0.6504`

### SALF-CBM with `resnet18_cub`

Run description:
- Model: `SALF-CBM`
- Dataset: `CUB`
- Concept bank: `concept_files/cub_filtered.txt` from upstream `VLG-CBM`
- Backbone: `resnet18_cub`
- CLIP pseudo-label model: `ViT-B/16`
- Spatial target source: `prompt_grid`
- Grid: `7 x 7`
- Prompt radius: `32`

Checkpoint and artifact directory:
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_02_03_07_28`

Run metadata and logs:
- Pod train log: `/workspace/SAVLGCBM/logs/cub_salf_resnet18_cub_pod2_small.log`
- Pod NEC log: `/workspace/SAVLGCBM/logs/cub_salf_resnet18_cub_nec_fast.log`
- Local train log copy: `logs/cub_salf_resnet18_cub_pod2_small.log`
- Local NEC log copy: `logs/cub_salf_resnet18_cub_nec_fast.log`

Full-accuracy result files:
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_02_03_07_28/train_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_02_03_07_28/val_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_02_03_07_28/test_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_02_03_07_28/metrics.txt`

Sparse / NEC result files:
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_02_03_07_28/metrics.csv`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_02_03_07_28/W_g@NEC=5.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_02_03_07_28/W_g@NEC=10.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_02_03_07_28/W_g@NEC=15.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_02_03_07_28/W_g@NEC=20.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_02_03_07_28/W_g@NEC=25.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_02_03_07_28/W_g@NEC=30.pt`

Recorded metrics:
- Train accuracy: `1.0000`
- Val accuracy: `0.7083`
- Test accuracy: `0.7083`
- `ACC@5`: `0.4271`
- `ACC@10`: `0.6368`
- `ACC@15`: `0.6812`
- `ACC@20`: `0.6900`
- `ACC@25`: `0.6964`
- `ACC@30`: `0.7009`
- `AVGACC`: `0.6387`

Current interpretation:
- The earlier `SALF-CBM` `clip_RN50` `CUB` run was severely underperforming and is best treated as a protocol mismatch.
- Under the more paper-aligned `resnet18_cub` backbone, `SALF-CBM` recovers to a much more competitive regime.
- On these reduced-scope NEC reruns, `SALF-CBM resnet18_cub` is very close to `LF-CBM resnet18_cub` and clearly much stronger than the earlier `SALF-CBM clip_RN50` run.

### SALF-CBM with `resnet18_cub`, original LF bank, and fixed LF similarity

Run description:
- Model: `SALF-CBM`
- Dataset: `CUB`
- Concept bank: `concept_files/label_free_cbm_cub_filtered.txt`
- Backbone: `resnet18_cub`
- CLIP pseudo-label model: `ViT-B/16`
- Spatial target source: `prompt_grid`
- Protocol: `lf_original_protocol=true`
- Key fixes:
  - exact upstream LF cubed-cosine similarity restored for SALF losses
  - raw LF concept loading preserved
  - prompt-grid extraction tensorized to remove the slow PIL loop

Checkpoint and artifact directory:
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_03_05_30_10`

Run metadata and logs:
- Pod train log: `/workspace/SAVLGCBM/logs/cub_salf_resnet18_cub_lfbank_fixedsim.log`

Full-accuracy result files:
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_03_05_30_10/train_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_03_05_30_10/val_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_03_05_30_10/test_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_03_05_30_10/concept_layer.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_03_05_30_10/W_g.pt`
- `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_03_05_30_10/b_g.pt`

Recorded dense metrics:
- Train accuracy: `1.0000`
- Val accuracy: `0.7320`
- Test accuracy: `0.7320`

Recorded tuned NEC metrics:
- Sweep settings: `--lam 0.01 --max_glm_steps 40`
- `ACC@5`: `0.5335`
- `ACC@10`: `0.6938`
- `ACC@15`: `0.7159`
- `ACC@20`: `0.7174`
- `ACC@25`: `0.7174`
- `ACC@30`: `0.7174`
- `AVGACC`: `0.6826`

Current interpretation:
- This corrected SALF run is materially better than the earlier `resnet18_cub` fast-rerun result (`0.7083` dense, `0.6387` `AVGACC`).
- On sparse performance, it is now essentially tied with the corrected `LF-CBM` run:
  - `SALF-CBM AVGACC = 0.6826`
  - `LF-CBM AVGACC = 0.6817`
- It is still below `VLG-CBM` and the best current `SAVLG-CBM` run on `CUB`.

### SAVLG-CBM with `resnet18_cub`

Run description:
- Model: `SAVLG-CBM`
- Dataset: `CUB`
- Concept bank: `concept_files/cub_filtered.txt` from upstream `VLG-CBM`
- Backbone: `resnet18_cub`
- Spatial supervision: VLG annotation JSON logits for concept presence plus rasterized annotation boxes on a `7 x 7` patch grid

Checkpoint and artifact directory:
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04`

Run metadata and logs:
- Pod train log: `/workspace/SAVLGCBM/logs/cub_savlg_tmp.log`
- Pod NEC log: `/workspace/SAVLGCBM/logs/cub_savlg_nec_fast_rerun.log`

Full-accuracy result files:
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/train_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/val_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/test_metrics.json`
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/metrics.txt`

Sparse / NEC result files:
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/metrics.csv`
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/W_g@NEC=5.pt`
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/W_g@NEC=10.pt`
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/W_g@NEC=15.pt`
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/W_g@NEC=20.pt`
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/W_g@NEC=25.pt`
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/W_g@NEC=30.pt`
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/b_g@NEC=5.pt`
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/b_g@NEC=10.pt`
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/b_g@NEC=15.pt`
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/b_g@NEC=20.pt`
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/b_g@NEC=25.pt`
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_20_07_04/b_g@NEC=30.pt`

Recorded metrics:
- Train accuracy: `1.0000`
- Val accuracy: `1.0000`
- Test accuracy: `0.7459`
- `ACC@5`: `0.6917`
- `ACC@10`: `0.7150`
- `ACC@15`: `0.7218`
- `ACC@20`: `0.7230`
- `ACC@25`: `0.7244`
- `ACC@30`: `0.7259`
- `AVGACC`: `0.7170`

Comparison to `VLG-CBM` on the same backbone:
- `SAVLG-CBM` is below `VLG-CBM resnet18_cub` in dense accuracy: `0.7459` vs `0.7594`
- `SAVLG-CBM` is also below `VLG-CBM resnet18_cub` in sparse performance: `AVGACC 0.7170` vs `0.7556`
- The gap is modest in dense accuracy and larger in NEC:
  - dense delta: `-0.0135`
  - `AVGACC` delta: `-0.0386`

### SAVLG-CBM with `resnet18_cub` and Dice-enabled local loss

Run description:
- Model: `SAVLG-CBM`
- Dataset: `CUB`
- Backbone: `resnet18_cub`
- Loss variant: global presence BCE + patch BCE + Dice, with `loss_dice_w=0.25`

Checkpoint and artifact directory:
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_02_21_27_12`

Run metadata and logs:
- Pod train log: `/workspace/SAVLGCBM/logs/cub_savlg_dice.log`
- Pod NEC log: `/workspace/SAVLGCBM/logs/cub_savlg_dice_nec_fast80.log`

Recorded dense metrics:
- Train accuracy: `1.0000`
- Val accuracy: `1.0000`
- Test accuracy: `0.7456`

Recorded fast NEC metrics:
- `ACC@5`: `0.6926`
- `ACC@10`: `0.7114`
- `ACC@15`: `0.7114`
- `ACC@20`: `0.7114`
- `ACC@25`: `0.7114`
- `ACC@30`: `0.7114`
- `AVGACC`: `0.7083`

Comparison to the earlier non-Dice SAVLG baseline:
- Dense test accuracy changed from `0.7459` to `0.7456`
- `AVGACC` changed from `0.7170` to `0.7083`
- So this Dice setting was effectively neutral in dense accuracy and slightly worse in sparse performance

### SAVLG-CBM multiscale dual-branch local-MIL on `resnet18_cub`

Run description:
- Model: `SAVLG-CBM`
- Dataset: `CUB`
- Backbone: `resnet18_cub`
- This is the current working `SAVLG` architecture.
- Global path:
  - `conv5` feature map: `B x 512 x 7 x 7`
  - separate global `1 x 1` concept layer
  - global concept logits from average pooling over `7 x 7`
  - binary-threshold global concept targets matched to the `VLG-CBM` semantics
- Spatial path:
  - `conv4` feature map: `B x 256 x 14 x 14`
  - `conv5` projection: `1 x 1, 512 -> 512`, then bilinear upsample to `14 x 14`
  - `conv4` projection: `1 x 1, 256 -> 512`
  - fused feature: `ReLU(conv4_proj(conv4) + upsample(conv5_proj(conv5)))`
  - separate spatial `1 x 1` concept layer over the fused `14 x 14` map
  - `soft_box` local targets on a `14 x 14` supervision grid
  - containment local loss
  - local-MIL auxiliary loss with `lse` pooling
- Final classifier:
  - pooled global concept logits are standardized with the saved train mean/std
  - the sparse CBM final layer is fit on those global concept features
  - the classifier is still anchored to the global branch, not the spatial branch directly

Checkpoint and artifact directory:
- `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_26_28`

Run metadata and logs:
- Dense train log:
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_v1.log`
- NEC log:
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_v1_nec150.log`
- Native localization log:
  - `/workspace/SAVLGCBM/logs/native_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full_relaunch_pod2.log`
- Grad-CAM localization log:
  - `/workspace/SAVLGCBM/logs/gradcam_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full_relaunch_pod2.log`

Recorded dense metrics:
- Test accuracy: `0.7506044905008635`

Recorded tuned NEC metrics:
- `ACC@5`: `0.7445595860481262`
- `ACC@10`: `0.7482`
- `ACC@15`: `0.7491`
- `ACC@20`: `0.7485`
- `ACC@25`: `0.7487`
- `ACC@30`: `0.7489`
- `AVGACC`: `0.7479850351810455`

Recorded localization metrics:
- Native maps:
  - output:
    - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full_meanthr_bs32_nw8_nocache.json`
  - mean IoU `0.11942165039707688`
  - `mAP@0.3 = 0.47732729696210124`
  - `mAP@0.5 = 0.49152539540216145`
  - `mAP@0.7 = 0.49754218937165234`
  - point hit rate `0.9707708962452282`
  - point coverage `0.2525432752492913`
- Grad-CAM maps:
  - output:
    - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full.json`
  - mean IoU `0.07955921454802446`
  - `mAP@0.3 = 0.026529112111867603`
  - `mAP@0.5 = 0.010961884166584716`
  - `mAP@0.7 = 0.006548821727890776`
  - point hit rate `0.7189542483660131`
  - point coverage `0.011007669395801258`

Current interpretation:
- This is the best fully measured `SAVLG-CBM` checkpoint in the repo so far on classification:
  - dense improves over the earlier clean-global best `0.7492`
  - `ACC@5` improves over `0.7440`
  - `AVGACC` improves over `0.7479`
- Native-map localization improves substantially and is the strongest `SAVLG` localization evidence recovered so far in this repo.
- Grad-CAM localization remains weak, especially on the paper-style point metric, so this is not yet a full localization win over corrected `SALF` and still does not beat the current `VLG-CBM` frontier overall.
- The main architectural lesson is:
  - keep classification on the strong `conv5` global concept path
  - improve localization through a separate higher-resolution spatial branch
  - use softer local supervision instead of hard patch-IoU targets

## Active VLG-warm Local-loss Ablation Snapshot

Current `TASK.md` line:
- backbone: `resnet18_cub`
- warm-start global concept head from `VLG-CBM`
- freeze global branch initially
- frozen backbone
- multiscale spatial branch (`conv4 + upsample(conv5)` to `14 x 14`)
- local-loss ablations only

Completed dense ablations on `atharv-rwx-pod-2`:
- all six allowed warm-start frozen-global multiscale local-loss ablations converged to the same dense test accuracy:
  - dense test accuracy `0.7590673575129534`
- completed dense checkpoints:
  - containment only:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_03_54_03`
  - local-mil only:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_03_54_15`
  - soft-align only:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_07_57`
  - soft-align + local-mil:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_07_57-1`
  - soft-align + outside penalty:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13`
  - soft-align + local-mil + outside penalty:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13-1`

Completed sparse follow-up on all six dense-qualified variants:
- `containment only`
  - log:
    - `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_containment_only_v1_nec150.log`
  - `ACC@5 = 0.7549`
  - `AVGACC = 0.7560`
- `local-mil only`
  - log:
    - `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_localmil_only_v1_nec150.log`
  - `ACC@5 = 0.7549`
  - `AVGACC = 0.7560`
- `soft-align only`
  - log:
    - `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_only_v1_nec150.log`
  - `ACC@5 = 0.7549`
  - `AVGACC = 0.7560`
- `soft-align + local-mil`
  - log:
    - `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_localmil_v1_nec150.log`
  - `ACC@5 = 0.7549`
  - `AVGACC = 0.7560`
- `soft-align + outside penalty`
  - log:
    - `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_outside_v1_nec150.log`
  - `ACC@5 = 0.7549`
  - `AVGACC = 0.7560`
- `soft-align + local-mil + outside penalty`
  - log:
    - `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_localmil_outside_v1_nec150.log`
  - `ACC@5 = 0.7549`
  - `AVGACC = 0.7560`

Completed native GT-present localization on the four soft-align variants:
- `soft-align only`
  - log:
    - `/workspace/SAVLGCBM/logs/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_only_v1_full_meanthr_bs32_nw8_gtpresent.log`
  - output:
    - `/workspace/SAVLGCBM/results/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_only_v1_full_meanthr_bs32_nw8_gtpresent.json`
  - mean IoU `0.124099317393116`
  - `mAP@0.3 = 0.04719790371291147`
  - `mAP@0.5 = 0.032559621861637134`
  - `mAP@0.7 = 0.023469499207264884`
  - point hit `0.9763696831787152`
  - coverage `0.08220498726563737`
- `soft-align + local-mil`
  - log:
    - `/workspace/SAVLGCBM/logs/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_localmil_v1_full_meanthr_bs32_nw8_gtpresent.log`
  - output:
    - `/workspace/SAVLGCBM/results/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_localmil_v1_full_meanthr_bs32_nw8_gtpresent.json`
  - mean IoU `0.1250153356970898`
  - `mAP@0.3 = 0.0296853607373322`
  - `mAP@0.5 = 0.02518812710654583`
  - `mAP@0.7 = 0.02234245708810698`
  - point hit `0.9831700042607584`
  - coverage `0.06754248384822366`
- `soft-align + outside penalty`
  - log:
    - `/workspace/SAVLGCBM/logs/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_outside_v1_full_meanthr_bs32_nw8_gtpresent.log`
  - output:
    - `/workspace/SAVLGCBM/results/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_outside_v1_full_meanthr_bs32_nw8_gtpresent.json`
  - mean IoU `0.12455332332837685`
  - `mAP@0.3 = 0.04907096879397959`
  - `mAP@0.5 = 0.033569865976427096`
  - `mAP@0.7 = 0.023689953026913697`
  - point hit `0.9750421585160203`
  - coverage `0.08532742420536138`
- `soft-align + local-mil + outside penalty`
  - log:
    - `/workspace/SAVLGCBM/logs/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_localmil_outside_v1_full_meanthr_bs32_nw8_gtpresent.log`
  - output:
    - `/workspace/SAVLGCBM/results/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_localmil_outside_v1_full_meanthr_bs32_nw8_gtpresent.json`
  - mean IoU `0.1255463541960814`
  - `mAP@0.3 = 0.031103271337829037`
  - `mAP@0.5 = 0.025754280184576945`
  - `mAP@0.7 = 0.022404031878906265`
  - point hit `0.9838308457711443`
  - coverage `0.0694130681899938`

Completed native GT-present localization on both control variants:
- `containment only`
  - log:
    - `/workspace/SAVLGCBM/logs/native_savlg_vlgwarm_frozenglobal_multiscale_containment_only_v1_full_meanthr_bs32_nw8_gtpresent.log`
  - output:
    - `/workspace/SAVLGCBM/results/native_savlg_vlgwarm_frozenglobal_multiscale_containment_only_v1_full_meanthr_bs32_nw8_gtpresent.json`
  - mean IoU `0.12015226059184847`
  - `mAP@0.3 = 0.043528346568986145`
  - `mAP@0.5 = 0.032490715726793476`
  - `mAP@0.7 = 0.024530855580414665`
  - point hit `0.9673325010403662`
  - coverage `0.06915406420421025`
- `local-mil only`
  - log:
    - `/workspace/SAVLGCBM/logs/native_savlg_vlgwarm_frozenglobal_multiscale_localmil_only_v1_full_meanthr_bs32_nw8_gtpresent.log`
  - output:
    - `/workspace/SAVLGCBM/results/native_savlg_vlgwarm_frozenglobal_multiscale_localmil_only_v1_full_meanthr_bs32_nw8_gtpresent.json`
  - mean IoU `0.06317313391705029`
  - `mAP@0.3 = 0.024668750019905216`
  - `mAP@0.5 = 0.02298420380038701`
  - `mAP@0.7 = 0.022159182919480994`
  - point hit `0.8277669902912621`
  - coverage `0.07410391815474049`

Current interpretation:
- the loss ablation has not moved dense classification at all so far; every allowed dense run lands at the same `0.7591` frontier
- sparse recovery is also flat across the full six-run local-loss ablation:
  - `containment only`, `local-mil only`, `soft-align only`, `soft-align + local-mil`, `soft-align + outside penalty`, and `soft-align + local-mil + outside penalty` all land at `ACC@5 = 0.7549` and `AVGACC = 0.7560`
- all four localized soft-align variants improve over the earlier warm-start proof-of-concept native checkpoint (`mean IoU 0.1198`, `mAP@0.3 0.0284`, `mAP@0.5 0.0260`, `mAP@0.7 0.0228`, `point hit 0.9665`, `coverage 0.0649`)
- the finalized `containment only` control native result is competitive with the soft-align variants and is best overall on `mAP@0.7`, but it still trails the best soft-align variants on mean IoU, `mAP@0.3`, `mAP@0.5`, point hit, and coverage
- the finalized `local-mil only` control native result is materially worse than every box-aware loss (`containment` and all four soft-align variants) on mean IoU, `mAP`, and point hit, so plain MIL pooling alone is not competitive in this warm-start frozen-global regime
- `soft-align + outside penalty` is currently best on bbox `mAP@0.3`, `mAP@0.5`, and point coverage
- `soft-align + local-mil + outside penalty` is currently best on mean IoU and point hit
- this completes the six-run warm-start frozen-global local-loss ablation sprint under `TASK.md`; no eligible dense, sparse, or native-localization jobs remain in this line

### Residual spatial coupling sprint status

- Finalized dense and `NEC150` results for the three exact `TASK.md` residual runs:

| Variant | Dense acc | ACC@5 | ACC@10 | ACC@15 | ACC@20 | ACC@25 | ACC@30 | AVGACC | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Residual-Alpha-0.05` | `0.7592` | `0.7560` | `0.7568` | `0.7568` | `0.7570` | `0.7570` | `0.7572` | `0.7568` | Native done: IoU `0.1309`, `mAP@0.5` `0.0656`, hit `0.9889`, cov `0.3657` |
| `Residual-Alpha-0.10` | `0.7585` | `0.7572` | `0.7584` | `0.7580` | `0.7579` | `0.7584` | `0.7585` | `0.7581` | Native done: IoU `0.1307`, `mAP@0.5` `0.0657`, hit `0.9889`, cov `0.3651` |
| `Residual-Alpha-0.05-PlusMIL` | `0.7591` | `0.7563` | `0.7572` | `0.7572` | `0.7573` | `0.7584` | `0.7584` | `0.7575` | Native done: IoU `0.1216`, `mAP@0.5` `0.0434`, hit `0.9847`, cov `0.3014` |

- Exact pod artifact paths:
  - `Residual-Alpha-0.05`
    - dense run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_23_45_15-1`
    - NEC log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_residual_alpha005_softalign_outside_v1_nec150.log`
    - native log: `/workspace/SAVLGCBM/logs/native_savlg_residual_alpha005_softalign_outside_v1_full.log`
    - native output: `/workspace/SAVLGCBM/results/native_savlg_residual_alpha005_softalign_outside_v1_full_meanthr_bs32_nw8_gtpresent.json`
  - `Residual-Alpha-0.10`
    - dense run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_23_48_21`
    - NEC log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_residual_alpha010_softalign_outside_v1_nec150.log`
    - native output: `/workspace/SAVLGCBM/results/native_savlg_residual_alpha010_softalign_outside_v1_full_meanthr_bs32_nw8_gtpresent.json`
  - `Residual-Alpha-0.05-PlusMIL`
    - dense run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_08_00_03_22`
    - NEC log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_residual_alpha005_softalign_localmil_outside_v1_nec150.log`
    - native output: `/workspace/SAVLGCBM/results/native_savlg_residual_alpha005_localmil_outside_v1_full_meanthr_bs32_nw8_gtpresent.json`

- Finalized native GT-present localization:
  - `Residual-Alpha-0.05`
    - mean IoU `0.13085882710717972`
    - `mAP@0.3 = 0.1188843234267777`
    - `mAP@0.5 = 0.0656205807861721`
    - `mAP@0.7 = 0.03112899065322384`
    - point hit `0.9889426671388659`
    - coverage `0.3656704605954214`
  - `Residual-Alpha-0.10`
    - mean IoU `0.13070872910587703`
    - `mAP@0.3 = 0.1188683369104664`
    - `mAP@0.5 = 0.06573367546834702`
    - `mAP@0.7 = 0.030992899959122733`
    - point hit `0.9889265447667087`
    - coverage `0.36513806351353295`
  - `Residual-Alpha-0.05-PlusMIL`
    - mean IoU `0.12157116163951714`
    - `mAP@0.3 = 0.07900841114248819`
    - `mAP@0.5 = 0.043425785090992264`
    - `mAP@0.7 = 0.024953646221050316`
    - point hit `0.9846741465743615`
    - coverage `0.30137991567981354`

- Current interpretation:
  - residual coupling preserves dense performance essentially at the `VLG-CBM resnet18_cub` level for all three exact runs
  - both plain residual variants materially improve over the frozen-global `soft-align + outside penalty` localization baseline on mean IoU, all three reported `mAP` thresholds, point hit, and coverage
  - `Residual-Alpha-0.05` is the best dense checkpoint and is also slightly best on native mean IoU, `mAP@0.3`, `mAP@0.7`, point hit, and coverage
  - `Residual-Alpha-0.10` is slightly best on sparse recovery (`ACC@5`, `AVGACC`) and by a negligible margin on native `mAP@0.5`
  - `Residual-Alpha-0.05-PlusMIL` does not improve on either plain residual variant, so `+local_mil` is not justified in this stage
  - this sprint is complete and establishes residual coupling as the first residual-stage line that preserves `VLG`-level dense behavior while also materially improving native localization over the frozen-global outside-penalty baseline

### Spatial-Source Residual Ablation (`alpha=0.05`, `soft-align + outside`, VLG warm-start)

This ablation holds the residual recipe fixed and varies only the spatial branch source.

| Variant | Dense acc | ACC@5 | ACC@10 | ACC@15 | ACC@20 | ACC@25 | ACC@30 | AVGACC | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `conv4+conv5` | `0.7592` | `0.7560` | `0.7568` | `0.7568` | `0.7572` | `0.7570` | `0.7572` | `0.7568` | Current multiscale residual baseline |
| `conv5-only` | `0.7592` | `0.7560` | `0.7566` | `0.7565` | `0.7568` | `0.7570` | `0.7572` | `0.7567` | Matches multiscale on dense/sparse |
| `conv4-only` | `0.7587` | `0.7547` | `0.7558` | `0.7561` | `0.7558` | `0.7563` | `0.7566` | `0.7559` | Slightly worse than `conv5-only` |
| `conv3-only` | `0.7587` | `0.7549` | `0.7558` | `0.7561` | `0.7558` | `0.7561` | `0.7566` | `0.7559` | Essentially tied with `conv4-only` |

- Exact artifact paths:
  - `conv4+conv5`
    - dense / NEC run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_23_45_15-1`
  - `conv5-only`
    - dense / NEC run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_08_02_43_40`
    - NEC log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_residual_alpha005_softalign_outside_conv5only_v1_nec150.log`
  - `conv4-only`
    - dense / NEC run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_08_02_43_40-1`
    - NEC log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_residual_alpha005_softalign_outside_conv4only_v1_nec150.log`
  - `conv3-only`
    - dense / NEC run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_08_02_57_19`
    - NEC log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_residual_alpha005_softalign_outside_conv3only_v1_nec150.log`

- Interpretation:
  - `conv5-only` carries essentially all of the dense and low-NEC benefit.
  - `conv4+conv5` is not justified by classification or sparse performance alone.
  - `conv4-only` and `conv3-only` are both slightly weaker and are not competitive replacements for `conv5`.
  - The remaining reason to keep `conv4+conv5` would have to be better localization, not better dense / NEC behavior.

- Native GT-present localization for the same spatial-source ablation:
  - `conv5-only`
    - mean IoU `0.1251`
    - `mAP@0.3 = 0.1168`
    - `mAP@0.5 = 0.0630`
    - `mAP@0.7 = 0.0312`
    - point hit `0.9711`
    - coverage `0.3583`
  - `conv4-only`
    - mean IoU `0.1445`
    - `mAP@0.3 = 0.1170`
    - `mAP@0.5 = 0.0710`
    - `mAP@0.7 = 0.0395`
    - point hit `0.9858`
    - coverage `0.3370`
  - `conv3-only`
    - mean IoU `0.1113`
    - `mAP@0.3 = 0.0906`
    - `mAP@0.5 = 0.0488`
    - `mAP@0.7 = 0.0323`
    - point hit `0.9147`
    - coverage `0.3275`

- Localization interpretation:
  - `conv4-only` localizes best on mean IoU, `mAP@0.5`, `mAP@0.7`, and point hit.
  - `conv5-only` remains strong and has the best coverage.
  - `conv3-only` is clearly weakest and should not be promoted.

### Residual Alpha Sweep (`conv5-only`, `soft-align + outside`, VLG warm-start)

This sweep holds the successful residual recipe fixed and varies only the coupling strength in

`c_final = c_global + alpha * c_spatial`.

| Alpha | Dense acc | AVGACC | Status |
| --- | ---: | ---: | --- |
| `0.00` | `0.7591` | `0.7556` | Pure VLG anchor with no spatial contribution |
| `0.02` | `0.7589` | `0.7561` | Slight sparse improvement, slight dense regression |
| `0.10` | `0.7592` | `0.7585` | Best sparse tier |
| `0.20` | `0.7606` | `0.7585` | Best overall tradeoff |
| `0.30` | `0.7606` | `0.7580` | Dense-best tier, slightly weaker sparse |
| `0.40` | `0.7594` | `0.7585` | Best sparse tier, dense back to VLG |

- Interpretation:
  - `alpha=0.00` and `0.02` underuse the spatial branch and leave sparse gains on the table.
  - `alpha=0.10`, `0.20`, and `0.40` all reach the top sparse tier.
  - `alpha=0.20` is the current best overall choice because it sits in both the dense-best tier and the sparse-best tier.
  - `alpha=0.30` is dense-best but no longer sparse-best.

### Unified Grad-CAM Localization Comparison (`gt_present`, `mean` threshold)

These runs use the same Grad-CAM localization protocol across:
- `VLG-CBM`
- `LF-CBM`
- `SALF-CBM`
- best residual `SAVLG-CBM` (`alpha=0.20`)

Artifacts:
- `VLG`: `/workspace/SAVLGCBM/results/gradcam_vlg_cub_gtpresent.json`
- `LF`: `/workspace/SAVLGCBM/results/gradcam_lf_cub_gtpresent.json`
- `SALF`: `/workspace/SAVLGCBM/results/gradcam_salf_cub_gtpresent.json`
- `SAVLG`: `/workspace/SAVLGCBM/results/gradcam_savlg_alpha020_cub_gtpresent.json`

| Model | mean IoU | mAP@0.3 | mAP@0.5 | mAP@0.7 | point hit | coverage | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `VLG-CBM` | `0.0874` | `0.1054` | `0.0491` | `0.0288` | `0.8000` | `0.2536` | Strong post-hoc baseline; best coverage among non-spatial models |
| `LF-CBM` | `0.0750` | `0.0366` | `0.0361` | `0.0364` | `0.5626` | `0.0334` | Weak overall; only slight edge on `mAP@0.7` |
| `SALF-CBM` | `0.0789` | `0.0306` | `0.0295` | `0.0287` | `0.7284` | `0.0446` | Below `VLG` and `SAVLG` on Grad-CAM localization |
| `SAVLG-CBM` (`alpha=0.20`) | `0.1062` | `0.1174` | `0.0546` | `0.0313` | `0.9011` | `0.1341` | Best overall Grad-CAM localization |

- Interpretation:
  - `SAVLG-CBM` is the strongest overall model on Grad-CAM localization.
  - `VLG-CBM` remains the strongest non-spatial baseline and still has the best coverage.

### ResNet-50 MMPretrain CUB Baseline

We integrated the official MMPretrain `resnet50_8xb8_cub` checkpoint as backbone `resnet50_cub_mm` and first trained a plain `VLG-CBM` baseline on top of it.

Finished checkpoint:
- `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_08_19_02_52`

Default sparse final-layer checkpoint:
- `saga_lam = 0.0002`
- `test accuracy = 0.85285`

Fast NEC sweep:
- log: `/workspace/SAVLGCBM/logs/cub_vlg_cbm_resnet50_cub_mm_nec150_fast.log`
- runtime: `26m 59.9s`
- `ACC@5 = 0.8501`
- `ACC@10 = 0.8522`
- `ACC@15 = 0.8541`
- `ACC@20 = 0.8546`
- `ACC@25 = 0.8546`
- `ACC@30 = 0.8546`
- `AVGACC = 0.8533`

- Interpretation:
  - The MMPretrain ResNet-50 CUB backbone materially raises the CBM ceiling over the earlier `resnet18_cub` line.
  - Sparse performance stays extremely close to the default `saga_lam = 0.0002` checkpoint accuracy, so low-NEC degradation is much smaller than on the weaker backbone.
  - `SALF-CBM` does not beat `VLG-CBM` on this post-hoc localization view.
  - `LF-CBM` is weakest overall despite a small edge on `mAP@0.7`.

### ResNet-50 SAVLG branch/loss ablations

Completed ResNet-50 SAVLG runs:

| Run | Full Acc | ACC@5 | AVGACC | MaxBoxAcc@0.3 | MaxBoxAcc@0.5 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `conv5 + cov025` | `0.8566` | `0.8558` | `0.8564` | `0.3301` | `0.1408` | baseline conv5 branch |
| `conv5 + cov050` | `0.8566` | `0.8544` | `0.8561` | `0.3207` | `0.1371` | stronger coverage did not help |
| `conv5 + dice010` | `0.8570` | `0.8558` | `0.8563` | `0.3393` | `0.1462` | best finished conv5 localization |
| `conv4` | `0.8584` | `0.8551` | `0.8559` | `0.3709` | `0.1573` | best dense accuracy in the 5-run sweep |
| `conv4+conv5` | `0.8573` | `0.8565` | `0.8561` | `0.4467` | `0.2152` | best localization |
| `conv5 + soft-align only` | `0.8615` | `0.8570` | `0.8568` | `pending` | `pending` | no outside penalty, no coverage, no Dice |

Additional ResNet-50 checkpoints / baselines:

| Run | Full Acc | ACC@5 | AVGACC | Notes |
| --- | ---: | ---: | ---: | --- |
| `SALF-CBM resnet50_cub_mm` | `0.7263` | `0.2936` | `0.5540` | training and NEC completed; localization pending |
| `SAVLG conv4 alpha sweep best` | `0.85838` | `N/A` | `N/A` | inference-time alpha `0.1` and `0.3` tied on full test accuracy |

Branch-calibration diagnostic for `conv4+conv5` (`alpha=0.2`, test split):
- mean abs global logit: `9.3416`
- mean abs spatial logit: `2.3233`
- mean abs scaled spatial logit: `0.4647`
- mean abs fused logit: `8.9688`
- sign agreement fraction: `0.1573`
- fraction where `|alpha * spatial| > |global|`: `0.0036`

Interpretation:
- `conv4` is best for dense performance.
- `conv4+conv5` is the clear localization winner.
- `conv5 + soft-align only` is the strongest completed conv5 classifier-side result so far.
- The branch-calibration diagnostic suggests the main issue is not spatial dominance but poor sign agreement between global and spatial logits.
