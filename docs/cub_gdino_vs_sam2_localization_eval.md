# CUB Localization Eval: GDINO Boxes vs GroundedSAM2 Masks (Avoiding Protocol Drift)

This doc is the "single source of truth" for how we evaluate SAVLG-CBM spatial maps on CUB when comparing:

- **GT source**: stored GroundingDINO concept boxes vs SAM2-refined concept masks (GroundedSAM2 artifacts)
- **Training supervision**: models trained using GDINO vs models trained using GroundedSAM2

The goal is to prevent wasting time chasing phantom regressions caused by *evaluation protocol drift*.

## Key Takeaways (What Went Wrong Before)

1. The ~`0.28` **mIoU** number came from **`map_normalization=concept_zscore_minmax`**.
   - Using `sigmoid` normalization yields much lower mIoU (roughly ~0.11-0.13 in our earlier checks) and is not comparable.
2. Use `--eval_subset_mode gt_present` for meaningful localization comparisons.
   - `all` mixes in many image/concept pairs with no GT for that concept in that image (metrics get diluted / become hard to interpret).
3. When running on pods, `DataLoader worker ... killed by signal: Killed` is usually a stability issue.
   - Fix by using `--num_workers 0` (most stable) and/or reducing `--batch_size`.
4. Some SAM2-run folders exist but are **incomplete** (missing `concepts.txt` / `concept_layer.pt`), so evaluation will fail.

## Canonical Command (Use This Unless You Have A Good Reason)

This is the protocol that matched the original "old sweep localization" numbers:

```bash
python -u scripts/evaluate_native_spatial_maps.py \
  --load_paths <CKPT_DIR> \
  --names <NAME> \
  --annotation_dir /workspace/SAVLGCBM/annotations \
  --dataset cub --split val --concept_mode intersection \
  --map_source native \
  --map_normalization concept_zscore_minmax \
  --activation_thresholds 0.3,0.5,0.7 \
  --box_iou_thresholds 0.3,0.5,0.7 \
  --gt_source <gdino_boxes|groundedsam2_masks> \
  --groundedsam2_manifest /workspace/SAVLGCBM-sam3/results/groundedsam2_from_annotations_cub_val_full_bs16/manifest.json \
  --eval_subset_mode gt_present \
  --gt_cache_max_entries 0 \
  --batch_size 512 --num_workers 2 \
  --output results/gdino_vs_sam2/<OUT>.json
```

Notes:
- Only pass `--groundedsam2_manifest ...` when `--gt_source groundedsam2_masks`.
- The evaluator reports metrics for each activation threshold; "best" metrics are obtained by taking the max over thresholds.

## Stable Pod Settings (Recommended)

If you see worker-killed errors, run:

```bash
--batch_size 128 --num_workers 0
```

This is slower per image but extremely stable and avoids hidden process-level OOMs.

## Where Results Live (Pod)

On the pod we used:
- Repo: `/workspace/SAVLGCBM-main`
- Output directory: `/workspace/SAVLGCBM-main/results/gdino_vs_sam2/`

Each eval writes:
- `<name>_...json` (machine-readable results)
- `<name>_...log` (tqdm progress + logs)

## Ground Truth Sources

### GDINO boxes
- `--gt_source gdino_boxes`
- Loads per-image concept boxes from the annotation store (e.g. `/workspace/SAVLGCBM/annotations/cub_val`).

### GroundedSAM2 masks
- `--gt_source groundedsam2_masks`
- Uses cached per-image bundles indexed by:
  - `/workspace/SAVLGCBM-sam3/results/groundedsam2_from_annotations_cub_val_full_bs16/manifest.json`

Important: GroundedSAM2 here is *not* live grounding; it is:
stored GDINO concept label + box -> SAM2 box-prompted mask refinement.

## Which Concepts Are Evaluated?

The evaluator prints something like:
`Evaluating ... on 5790 images with 671 shared annotated concepts`

That "671" is the **intersection** of:
- the model's concept list (`concepts.txt` in the checkpoint dir)
- the concepts that exist in the annotation store for the split

So you are not evaluating all concepts in `concept_files/...` automatically; you're evaluating the shared annotated set.

## The Exact Run That Produced ~0.28 mIoU

We got **best mIoU = 0.2815077839833656 @ act thr=0.7** from:

- Checkpoint: `r18_a0p6` (GDINO-trained)
- Eval protocol:
  - `--eval_subset_mode gt_present`
  - `--map_normalization concept_zscore_minmax`
  - `--gt_source gdino_boxes`

Saved JSON:
- `/workspace/SAVLGCBM-main/results/gdino_vs_sam2/r18_a0p6_gt_gdino_boxes_gtpresent_b512_zscoreminmax.json`

## Apples-to-Apples: GDINO-trained (Exact Cov0 Sweep)

All runs below used:
- `concept_zscore_minmax`
- `gt_present`
- best over `activation_thresholds=0.3,0.5,0.7`

Files (pod):
- `/workspace/SAVLGCBM-main/results/gdino_vs_sam2/r18_a0p2_gt_gdino_boxes_gtpresent_b512_zscoreminmax.json`
- `/workspace/SAVLGCBM-main/results/gdino_vs_sam2/r18_a0p2_gt_sam2_masks_gtpresent_b512_zscoreminmax.json`
- `/workspace/SAVLGCBM-main/results/gdino_vs_sam2/r18_a0p6_gt_gdino_boxes_gtpresent_b512_zscoreminmax.json`
- `/workspace/SAVLGCBM-main/results/gdino_vs_sam2/r18_a0p6_gt_sam2_masks_gtpresent_b512_zscoreminmax.json`

Summary:

| alpha | GT | best mIoU (act thr) | best mAP@0.3 (act thr) | best mAP@0.5 (act thr) |
|---:|---|---:|---:|---:|
| 0.2 | GDINO boxes | 0.2161 (0.7) | 0.2788 (0.7) | 0.1099 (0.7) |
| 0.2 | SAM2 masks | 0.1471 (0.7) | 0.1571 (0.7) | 0.0653 (0.7) |
| 0.6 | GDINO boxes | 0.2815 (0.7) | 0.3613 (0.7) | 0.1443 (0.7) |
| 0.6 | SAM2 masks | 0.1777 (0.7) | 0.1892 (0.7) | 0.0788 (0.7) |

## SAM2-trained Checkpoints (GroundedSAM2 Supervision)

These checkpoints have the correct artifacts (includes `concepts.txt`, `concept_layer.pt`):

- alpha=0.8:
  - `/workspace/SAVLGCBM/saved_models/savlg_cub_gsam2_a0p8_20260420_173000/savlg_cbm_cub_2026_04_20_17_32_20`
- alpha=1.0:
  - `/workspace/SAVLGCBM/saved_models/savlg_cub_gsam2_a1p0_20260420_173008/savlg_cbm_cub_2026_04_20_17_32_25`

Do not use this directory for eval (incomplete; missing `concepts.txt`):
- `/workspace/SAVLGCBM/saved_models/savlg_cub_gsam2_a0p8_20260420_172107/savlg_cbm_cub_2026_04_20_17_23_57`

### SAM2-trained alpha=0.8 (results)

Eval protocol: `gt_present`, `concept_zscore_minmax`.

Against GDINO boxes GT:
- best mIoU: **0.2336** @ act thr 0.7
- best mAP@0.3: **0.2894** @ act thr 0.7
- best mAP@0.5: **0.1073** @ act thr 0.7

Against SAM2 masks GT:
- best mIoU: **0.1578** @ act thr 0.7
- best mAP@0.3: **0.1627** @ act thr 0.7
- best mAP@0.5: **0.06233** @ act thr 0.7

Files:
- `/workspace/SAVLGCBM-main/results/gdino_vs_sam2/gsam2train_a0p8_gt_gdino_boxes_gtpresent_b128_zscoreminmax.json`
- `/workspace/SAVLGCBM-main/results/gdino_vs_sam2/gsam2train_a0p8_gt_sam2_masks_gtpresent_b128_zscoreminmax.json`

### Denser activation thresholds (thr grid)

Using `activation_thresholds=0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9`:

Against GDINO boxes GT:
- best mIoU: **0.23506** @ act thr 0.75
- best mAP@0.3: **0.33427** @ act thr 0.8
- best mAP@0.5: **0.12996** @ act thr 0.85

Against SAM2 masks GT:
- best mIoU: **0.15831** @ act thr 0.75
- best mAP@0.3: **0.16746** @ act thr 0.75
- best mAP@0.5: **0.06669** @ act thr 0.85

Files:
- `/workspace/SAVLGCBM-main/results/gdino_vs_sam2/gsam2train_a0p8_thrgrid_gt_gdino_boxes_gtpresent_b128_zscoreminmax.json`
- `/workspace/SAVLGCBM-main/results/gdino_vs_sam2/gsam2train_a0p8_thrgrid_gt_sam2_masks_gtpresent_b128_zscoreminmax.json`

## Direct Comparison: SAM2-trained vs GDINO-trained (alpha=0.8, eval vs GDINO-box GT)

Both evaluated with:
- `gt_present`
- `concept_zscore_minmax`
- `gt_source=gdino_boxes`

Results:
- SAM2-trained (`gsam2train_a0p8`): mIoU **0.2336**, mAP@0.3 **0.2894**, mAP@0.5 **0.1073**
- GDINO-trained (`gdinoTrain_a0p8`): mIoU **0.2550**, mAP@0.3 **0.3389**, mAP@0.5 **0.1298**

Delta (SAM2 - GDINO):
- mIoU: **-0.0214**
- mAP@0.3: **-0.0495**
- mAP@0.5: **-0.0225**

## Quick Parser (Extract Best Metrics from JSON)

```python
import json

def best_mean_iou(md):
    thr, val = max(((float(k), v) for k, v in md["mean_iou_by_activation_threshold"].items()),
                   key=lambda kv: kv[1])
    return val, thr

def best_map(md, box_iou):
    best = None
    for act, sub in md["map_by_activation_and_box_iou_threshold"].items():
        for k, v in sub.items():
            if abs(float(k) - box_iou) < 1e-9:
                if best is None or v > best[0]:
                    best = (v, float(act))
    return best

path = "results/gdino_vs_sam2/<file>.json"
d = json.load(open(path))
name = list(d["models"].keys())[0]
md = d["models"][name]
miou, miou_thr = best_mean_iou(md)
map03 = best_map(md, 0.3)
map05 = best_map(md, 0.5)
print(name, "mIoU", miou, "thr", miou_thr, "mAP@0.3", map03, "mAP@0.5", map05)
```

## Operational Notes (Pods)

- If your pod has an `activeDeadlineSeconds`, long evals can get killed mid-run.
- If you run multiple `nohup ... &` commands, verify you only have one eval per GPU:
  - `ps aux | grep evaluate_native_spatial_maps.py | grep -v grep`
- If ResNet18-CUB weights try to download and fail, stage the file to:
  - `/root/.torch/models/resnet18_cub-2333-200d8b9c.pth`

