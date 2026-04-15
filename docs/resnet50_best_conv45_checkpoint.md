# ResNet-50 Best `conv4+conv5` SAVLG Checkpoint

Checkpoint directory:

`/workspace/SAVLGCBM/saved_models/savlg_cbm_cub_2026_04_09_20_39_53`

Reproduced localization artifact:

`/workspace/SAVLGCBM/results/native_savlg_r50_c45_c25_gtpresent_zscore_sweep_exact_rerun.json`

Matched evaluation scheme:

- script: `scripts/evaluate_native_spatial_maps.py`
- `map_source = native`
- `map_normalization = concept_zscore_minmax`
- `threshold_mode = fixed`
- thresholds: `0.05, 0.10, ..., 0.95`
- `annotation_threshold = 0.15`
- `concept_mode = intersection`
- `eval_subset_mode = gt_present`
- `split = test`
- `interpolate_to_full_image = False`

Reproduced metrics:

- best `mean IoU = 0.2821824` at `thr = 0.85`
- best `mAP@0.3 = 0.4468902` at `thr = 0.9`
- best `mAP@0.5 = 0.2151663` at `thr = 0.9`
- best `mAP@0.7 = 0.0567413` at `thr = 0.9`
