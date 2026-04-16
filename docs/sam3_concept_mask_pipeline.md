# SAM3 Concept Pseudo-Mask Pipeline

This branch starts with a separate SAM3 concept-mask cache. The cache is
generated from CUB images and the CUB concept set directly. It does not use the
existing concept-to-part mapping as a gate.

## Minimal Architecture

1. `scripts/generate_sam3_concept_masks.py`
   - loads `concept_files/cub_filtered.txt` directly
   - loads CUB train/val images from the same `DATASET_FOLDER/CUB/{train,test}` layout used by `data.utils`
   - supports small image/concept subsets
   - writes a manifest plus per-record metadata in dry-run mode
   - can run a base SAM3 backend through `Sam3ConceptMaskRunner.predict`
   - records backend failures in the manifest when `continue_on_error=true`

2. `data/sam3_concept_mask_cache.py`
   - owns cache schema constants and path helpers
   - builds one record for each `(image_index, concept_index)` pair
   - stores concept indices in the original concept-set coordinate system

3. `scripts/render_sam3_concept_mask_cache.py`
   - reads a manifest
   - overlays existing `.npz` masks on the original image
   - skips planned/pending records that do not yet have mask files

## Cache Layout

Default root:

```text
saved_activations/sam3_concept_masks/
  cub_sam3_concept_masks_v1/
    train/
      manifest.json
      masks/
        image_000000/
          concept_0000.npz
      metadata/
        image_000000/
          concept_0000.json
      previews/
        image_000000/
          concept_0000.png
```

`manifest.json` uses schema `sam3_concept_masks_v1`:

```json
{
  "schema_version": "sam3_concept_masks_v1",
  "dataset": "cub",
  "split": "train",
  "concept_set": "concept_files/cub_filtered.txt",
  "concept_hash": "...",
  "selected_concept_indices": [0, 1, 2],
  "mask_format": "npz_bool_hxw_v1",
  "image_count": 8,
  "concept_count": 16,
  "record_count": 128,
  "records": [
    {
      "record_id": "train/000000/0000",
      "dataset_index": 0,
      "image_path": "...jpg",
      "image_size": [500, 375],
      "concept_index": 0,
      "concept": "long black bill",
      "prompt": "long black bill",
      "status": "planned",
      "score": null,
      "mask_format": "npz_bool_hxw_v1",
      "mask_path": "masks/image_000000/concept_0000.npz",
      "metadata_path": "metadata/image_000000/concept_0000.json",
      "preview_path": "previews/image_000000/concept_0000.png",
      "backend": "sam3",
      "failure_reason": null
    }
  ]
}
```

`concept_count` and `concept_hash` refer to the full loaded concept set. The
actual subset generated in a run is recorded by `selected_concept_indices` and
the manifest records. `concept_index` always stays in the original concept-set
coordinate system.

Each mask `.npz` should contain:

- `mask`: bool or uint8 array with shape `[image_h, image_w]`, in original image
  pixel coordinates.
- `score`: optional float confidence.
- `bbox_xyxy`: optional `[x1, y1, x2, y2]` pixel box.

The per-record metadata JSON mirrors the manifest record and records backend
settings. This keeps the binary mask file simple and easy to load.

## First Test Command

```bash
python scripts/generate_sam3_concept_masks.py \
  --config configs/sam3/cub_concept_masks_placeholder.json \
  --split train \
  --dry_run \
  --max_images 2 \
  --max_concepts 3 \
  --overwrite
```

This should create a small manifest with six planned records and no SAM3 model
dependency.

On the pod, use the real CUB root explicitly:

```bash
DATASET_FOLDER=/workspace/SAVLGCBM/datasets \
python scripts/generate_sam3_concept_masks.py \
  --config configs/sam3/cub_concept_masks_placeholder.json \
  --split train \
  --dry_run \
  --max_images 2 \
  --max_concepts 3 \
  --overwrite
```

## Current Status

The base-SAM3 configs in this branch are the intended run targets:

`configs/sam3/cub_concept_masks_base_20img_audit_pod.json`
`configs/sam3/cub_concept_masks_base_100img_3concept_pod.json`

The MedSAM3 LoRA configs remain in the tree only as deprecated comparison
artifacts. They are not referenced by the run scripts.

## Base SAM3 Backend

`configs/sam3/cub_concept_masks_base_100img_3concept_pod.json` points at the
base-SAM3 layout:

```text
/workspace/SAVLGCBM
/workspace/MedSAM3
Hugging Face checkpoint download for facebook/sam3
```

Run a tiny real-inference subset with:

```bash
DATASET_FOLDER=/workspace/SAVLGCBM/datasets \
python scripts/generate_sam3_concept_masks.py \
  --config configs/sam3/cub_concept_masks_base_100img_3concept_pod.json \
  --split train \
  --run \
  --max_images 100 \
  --max_concepts 3 \
  --overwrite
```

Use `SAM3_CHECKPOINT_PATH` to point at a different local checkpoint if needed;
otherwise the backend loads the base checkpoint from Hugging Face.

## SAVLG Training Hook

The existing SAVLG path already consumes:

- `global_concept_targets`: array shaped `[num_images, num_concepts]`
- `mask_entries`: list of dictionaries keyed by concept index, each value a
  patch-resolution mask shaped `[mask_h, mask_w]`
- `keep_idx`: concept indices retained for training

This branch adds a loader parallel to `methods.savlg.load_spatial_supervision`:

```text
load_sam3_spatial_supervision(manifest, raw_dataset, concepts, args, split_name)
```

That loader should:

1. read SAM3 pixel masks from the manifest
2. threshold `status == "ok"` and optional confidence score into global targets
3. downsample or rasterize pixel masks to `args.mask_h x args.mask_w`
4. emit `global_concept_targets`, `mask_entries`, and `keep_idx`

Training can switch source with:

```text
--savlg_supervision_source annotations|sam3_cache
--sam3_mask_manifest_train saved_activations/sam3_concept_masks/<cache>/train/manifest.json
--sam3_mask_manifest_val saved_activations/sam3_concept_masks/<cache>/val/manifest.json
--sam3_mask_downsample_mode area
--sam3_mask_binarize_threshold -1
```

The default remains `--savlg_supervision_source annotations`, so the existing
SAVLG training path is preserved unless the SAM3 cache source is selected.
