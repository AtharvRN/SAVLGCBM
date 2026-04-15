# CUB Part-Based Localization

This evaluation uses:

- the GPT-derived concept-to-part mapping in
  `results/cub_concept_part_mapping_gpt54.json`
- the official CUB part annotations in
  `CUB_200_2011/parts/parts.txt` and `CUB_200_2011/parts/part_locs.txt`
- the standard SAVLG spatial maps from a checkpoint

The evaluator is:

- [evaluate_savlg_cub_parts.py](/Users/atharvramesh/Projects/CBM/SAVLGCBM/scripts/evaluate_savlg_cub_parts.py)

## What It Measures

For each image:

1. read the GT concept annotations from `annotation_dir`
2. keep only concepts that the GPT mapping marked as part-aligned
3. map each kept concept to one or more exact CUB parts
4. look up visible CUB part points for that image
5. evaluate the concept spatial map against those part points

Primary metrics:

- `point_hit[r]`
  - whether the map peak falls within radius `r * image_diagonal` of any mapped part point
- `mean_normalized_distance`
  - minimum peak-to-part distance divided by image diagonal

Thresholded metrics:

- `point_in_mask`
- `mask_iou`
- `dice`
- `precision`
- `recall`
- `f1`
- pixel `tp/fp/fn`

The thresholded metrics compare the thresholded concept activation mask against a small disk around the mapped part point(s).

## Run

Use the `cbm` environment. Example:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate cbm

python -u /workspace/SAVLGCBM/scripts/evaluate_savlg_cub_parts.py \
  --load_path /workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_23_45_15-1 \
  --annotation_dir /workspace/SAVLGCBM/annotations \
  --cub_root /root/CUB_200_2011 \
  --mapping_json /workspace/SAVLGCBM/results/cub_concept_part_mapping_gpt54.json \
  --output /workspace/SAVLGCBM/results/cub_part_localization_conv45_old.json \
  --batch_size 128 \
  --num_workers 8 \
  --map_normalization concept_zscore_minmax \
  --point_source normalized_map \
  --activation_thresholds 0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --radius_fracs 0.01,0.02,0.05,0.1 \
  --disk_radius_frac 0.03
```

## Output

The JSON contains:

- `num_images`
- `num_gt_instances`
- `point_metrics.mean_normalized_distance`
- `point_metrics.point_hit`
- `threshold_metrics[thr]`
  - `point_in_mask`
  - `mask_iou`
  - `dice`
  - `pixel_counts.tp/fp/fn`
  - `precision`
  - `recall`
  - `f1`
- `best_mask_iou`
- `best_dice`
- `best_point_in_mask`

## Current Full-Split Results

These are the completed full-test runs using:

- GPT concept-to-part map:
  - [cub_concept_part_mapping_gpt54.json](/Users/atharvramesh/Projects/CBM/SAVLGCBM/results/cub_concept_part_mapping_gpt54.json)
- official CUB part points
- cached part-aligned annotation JSON:
  - `/root/cub_part_annotation_cache_gpt54.json`
- `map_normalization = concept_zscore_minmax`
- `point_source = normalized_map`
- thresholds:
  - `0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9`
- `disk_radius_frac = 0.03`

### ResNet-18 `conv4+conv5`

Result file:
- `/workspace/SAVLGCBM/results/cub_part_localization_r18_conv45_old_b256.json`

Summary:

| Metric | Value |
| --- | ---: |
| `num_images` | `5790` |
| `num_gt_instances` | `28531` |
| `mean_normalized_distance` | `0.5017` |
| `point_hit @ 0.01` | `0.0002` |
| `point_hit @ 0.02` | `0.0007` |
| `point_hit @ 0.05` | `0.0038` |
| `point_hit @ 0.10` | `0.0131` |
| `best point_in_mask` | `0.2387 @ thr=0.3` |
| `best mask_iou` | `0.0022893 @ thr=0.3` |
| `best dice` | `0.0045350 @ thr=0.3` |

At `thr=0.3`:

| Metric | Value |
| --- | ---: |
| `precision` | `0.0022757` |
| `recall` | `0.6525` |
| `f1` | `0.0045356` |

### ResNet-50 `conv4+conv5`

Result file:
- `/workspace/SAVLGCBM/results/cub_part_localization_r50_conv45_best_b256.json`

Summary:

| Metric | Value |
| --- | ---: |
| `num_images` | `5790` |
| `num_gt_instances` | `28531` |
| `mean_normalized_distance` | `0.1110` |
| `point_hit @ 0.01` | `0.0066` |
| `point_hit @ 0.02` | `0.0260` |
| `point_hit @ 0.05` | `0.1606` |
| `point_hit @ 0.10` | `0.5097` |
| `best point_in_mask` | `0.9960 @ thr=0.3` |
| `best mask_iou` | `0.0481050 @ thr=0.9` |
| `best dice` | `0.0861827 @ thr=0.9` |

At `thr=0.9`:

| Metric | Value |
| --- | ---: |
| `precision` | `0.0440` |
| `recall` | `0.4167` |
| `f1` | `0.0796` |

Interpretation:

- These are **part-point / part-disk localization metrics**, not the earlier concept-box localization metrics derived from GroundingDINO concept annotations.
- They are therefore **not directly comparable** to the earlier `mean IoU = 0.2822` native box-localization result for the same `resnet50 conv4+conv5` checkpoint.
- Under real CUB part supervision, the `resnet50` checkpoint is much stronger than `resnet18`, but both are substantially harder than the concept-box setting.

## Notes

- This is a real part-annotation evaluation, not the concept-box evaluator.
- It only scores concepts that the GPT mapping marked as aligned to a supported CUB part.
- Concepts without a mapped visible part in a given image are skipped.
