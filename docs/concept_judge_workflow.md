# Concept Judge Workflow

This workflow prepares **machine-judge / later human-judge** subsets for two distinct evaluations:

1. **Concept presence**
2. **Spatial faithfulness**

## Goal

For a fixed checkpoint, export:

- a subset of images
- the top positively contributing and positively activated concepts for each image
- optionally, the corresponding native concept-map overlays
- a judge task file for evaluating:
  - whether the concept is actually present in the image
  - and, for spatial models, whether the highlighted region corresponds to the concept

This is a **proxy evaluation** when full human labeling is not yet available.

## Export scripts

### 1. Concept presence

Use `scripts/export_concept_presence_subset.py`.

Supported checkpoints:

- `lf_cbm`
- `vlg_cbm`
- `salf_cbm`
- `savlg_cbm`

This export saves:

- original images
- top concept candidates
- judge tasks asking only whether the concept is present in the image

### 2. Spatial faithfulness

Use `scripts/export_spatial_judge_subset.py`.

Supported checkpoints:

- `salf_cbm`
- `savlg_cbm`

This export saves:

- original images
- native maps
- upsampled maps
- heatmap overlays
- judge tasks asking whether the highlighted region corresponds to the concept

## What the exports do

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
   - and for spatial export only:
     - native map (`.npy`)
     - upsampled normalized map (`.npy`)
     - heatmap overlay (`.png`)

The output also includes:

- `manifest.json`
- `judge_tasks.jsonl`
- `judge_prompt_template.txt`
- `judge_response_schema.json`

## Recommended first-pass exports

```bash
python scripts/export_concept_presence_subset.py \
  --load_path /path/to/checkpoint \
  --annotation_dir /path/to/annotations \
  --output_dir results/concept_presence_subset/example_run \
  --num_images 500 \
  --topk_concepts 5 \
  --class_source pred \
  --selection random
```

```bash
python scripts/export_spatial_judge_subset.py \
  --load_path /path/to/checkpoint \
  --annotation_dir /path/to/annotations \
  --output_dir results/spatial_judge_subset/example_run \
  --num_images 500 \
  --topk_concepts 5 \
  --class_source pred \
  --selection random \
  --map_normalization concept_zscore_minmax
```

## Output structure

```text
results/.../example_run/
  manifest.json
  judge_tasks.jsonl
  judge_prompt_template.txt
  judge_response_schema.json
  images/
  cases/

Spatial export additionally contains:

```text
  overlays/
  maps_native/
  maps_upsampled/
```
```

Each line in `judge_tasks.jsonl` is one concept-image pair and is the unit of machine or human judgment.

## Judge questions

For **concept presence** tasks, the judge should answer:

1. Is the concept visibly present in the image?

For **spatial faithfulness** tasks, the judge should answer:

1. Does the highlighted region actually correspond to that concept?

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

## OpenAI VLM judge runner

Use `scripts/run_concept_vlm_judge_openai.py` to consume either kind of `judge_tasks.jsonl` and write structured machine-judge outputs.

Example:

```bash
export OPENAI_API_KEY=...

python scripts/run_concept_vlm_judge_openai.py \
  --tasks-jsonl results/concept_judge_subset/example_run/judge_tasks.jsonl \
  --model gpt-5.4
```

Outputs:

- `judge_results_openai.jsonl`
- `judge_results_openai_summary.json`

The runner is resumable by `task_id`. Re-running it will skip tasks already written unless `--overwrite` is passed.

## Environment note

The `cbm` pod environment may not include the OpenAI package by default if the environment bootstrap skipped optional dependencies.

If needed:

```bash
conda activate cbm
pip install openai pydantic
```

You also need:

```bash
export OPENAI_API_KEY=...
```
