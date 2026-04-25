# Concept Judge Workflow

This workflow prepares a **machine-judge / later human-judge** subset for concept-level evaluation.

## Goal

For a fixed checkpoint, export:

- a subset of images
- the top positively contributing and positively activated concepts for each image
- the corresponding native concept-map overlays
- a judge task file for evaluating:
  - whether the concept is actually present in the image
  - whether the highlighted region corresponds to the concept

This is a **proxy evaluation** when full human labeling is not yet available.

## What the export script does

Use `scripts/export_concept_judge_subset.py`.

Supported checkpoints:

- `salf_cbm`
- `savlg_cbm`

For each selected image:

1. run the checkpoint on the test/val split
2. choose a reference class:
   - predicted class (`--class_source pred`), or
   - ground-truth class (`--class_source gt`)
3. compute per-concept class contribution:
   - `final_layer.weight[class] * normalized_concept_score`
4. keep concepts satisfying both:
   - positive contribution
   - positive activation
5. save, for each selected concept:
   - original image
   - native map (`.npy`)
   - upsampled normalized map (`.npy`)
   - heatmap overlay (`.png`)

The output also includes:

- `manifest.json`
- `judge_tasks.jsonl`
- `judge_prompt_template.txt`
- `judge_response_schema.json`

## Recommended first-pass export

```bash
python scripts/export_concept_judge_subset.py \
  --load_path /path/to/checkpoint \
  --annotation_dir /path/to/annotations \
  --output_dir results/concept_judge_subset/example_run \
  --num_images 500 \
  --topk_concepts 5 \
  --class_source pred \
  --selection random \
  --map_normalization concept_zscore_minmax
```

## Output structure

```text
results/concept_judge_subset/example_run/
  manifest.json
  judge_tasks.jsonl
  judge_prompt_template.txt
  judge_response_schema.json
  images/
  overlays/
  maps_native/
  maps_upsampled/
  cases/
```

Each line in `judge_tasks.jsonl` is one concept-image pair and is the unit of machine or human judgment.

## Judge questions

For each concept-image pair, the judge should answer:

1. Is the concept visibly present in the image?
2. Does the highlighted region actually correspond to that concept?

Allowed labels:

- concept presence: `yes`, `no`, `unsure`
- region match: `yes`, `partial`, `no`, `unsure`

## Important framing

This is **not** ground-truth concept accuracy.

It should be reported as one of:

- VLM-verified concept presence
- external VLM judge agreement
- proxy concept accuracy using a multimodal judge

If you later add human labels, use the same exported subset and task structure so the machine-judge and human-judge results remain directly comparable.
