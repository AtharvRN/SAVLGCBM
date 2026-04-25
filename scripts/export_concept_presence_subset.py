import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import torch
from loguru import logger
from scripts.evaluate_concept_interventions import (
    _build_test_loader,
    _get_batch_model_state,
    _lf_state,
    _load_checkpoint_args,
    _load_concepts,
    _resolve_num_images,
    _salf_state,
    _savlg_state,
    _vlg_state,
)
from data import utils as data_utils
from data.concept_dataset import _unwrap_dataset_indices
from methods.common import load_run_info
from tqdm import tqdm


JUDGE_PROMPT_TEMPLATE = """You are evaluating one concept-image pair for an interpretability study.

You will be shown:
1. the original image

Concept: {concept_name}
Model: {model_name}

Return a JSON object with exactly these fields:
- concept_present: one of ["yes", "no", "unsure"]
- concept_present_confidence: float in [0, 1]
- rationale_short: short string, max 30 words

Judging rules:
- concept_present should answer whether the concept is visibly present anywhere in the image.
- Use "unsure" when the concept is too subtle, ambiguous, occluded, or the image resolution is insufficient.
- Do not infer from class label alone. Judge only from visible evidence in the image.
"""


JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "concept_present",
        "concept_present_confidence",
        "rationale_short",
    ],
    "properties": {
        "concept_present": {
            "type": "string",
            "enum": ["yes", "no", "unsure"],
        },
        "concept_present_confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "rationale_short": {
            "type": "string",
        },
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export concept-presence judge tasks for LF/VLG/SALF/SAVLG checkpoints."
        )
    )
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--topk_concepts", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection",
        type=str,
        default="random",
        choices=["random", "first"],
    )
    parser.add_argument(
        "--class_source",
        type=str,
        default="pred",
        choices=["pred", "gt"],
    )
    parser.add_argument(
        "--min_positive_concepts",
        type=int,
        default=1,
    )
    return parser.parse_args()


def _slugify(raw: str) -> str:
    chars: List[str] = []
    for ch in raw.lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in {" ", "-", "_", "/"}:
            chars.append("_")
    slug = "".join(chars).strip("_")
    return slug or "item"


def _sample_positions(dataset_len: int, count: int, selection: str, seed: int) -> List[int]:
    keep = min(int(count), dataset_len)
    if selection == "first":
        return list(range(keep))
    rng = random.Random(seed)
    return sorted(rng.sample(range(dataset_len), keep))


def _get_final_layer(model_name: str, load_path: str, args):
    if model_name == "lf_cbm":
        final_layer = _lf_state(load_path, args).final
    elif model_name == "vlg_cbm":
        final_layer = _vlg_state(load_path, args)[3]
    elif model_name == "salf_cbm":
        final_layer = _salf_state(load_path, args)[4]
    elif model_name == "savlg_cbm":
        final_layer = _savlg_state(load_path, args)[4]
    else:
        raise NotImplementedError(model_name)
    final_layer.eval()
    return final_layer


def _write_static_judge_files(output_dir: Path) -> None:
    (output_dir / "judge_prompt_template.txt").write_text(JUDGE_PROMPT_TEMPLATE)
    (output_dir / "judge_response_schema.json").write_text(json.dumps(JUDGE_RESPONSE_SCHEMA, indent=2))


def main() -> None:
    os.chdir(Path(__file__).resolve().parents[1])
    cli_args = _parse_args()
    run_info = load_run_info(cli_args.load_path)
    model_name = run_info.get("model_name", run_info["args"]["model_name"])

    ckpt_args = _load_checkpoint_args(cli_args.load_path, cli_args.device, cli_args.annotation_dir)
    concepts = _load_concepts(cli_args.load_path)
    class_names = data_utils.get_classes(ckpt_args.dataset)

    requested_num_images = _resolve_num_images(cli_args)
    if requested_num_images is None:
        requested_num_images = int(cli_args.num_images)

    loader = _build_test_loader(
        model_name=model_name,
        load_path=cli_args.load_path,
        args=ckpt_args,
        concepts=concepts,
        batch_size_override=cli_args.batch_size,
        num_workers_override=cli_args.num_workers,
        num_images=None,
    )
    base_dataset, subset_indices = _unwrap_dataset_indices(loader.dataset)
    selected_positions = _sample_positions(
        dataset_len=len(loader.dataset),
        count=requested_num_images,
        selection=cli_args.selection,
        seed=cli_args.seed,
    )
    selected_pos_set = set(selected_positions)

    output_dir = Path(cli_args.output_dir)
    images_dir = output_dir / "images"
    cases_dir = output_dir / "cases"
    images_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)
    _write_static_judge_files(output_dir)

    final_layer = _get_final_layer(model_name, cli_args.load_path, ckpt_args)
    class_weights = final_layer.weight.detach().cpu()

    summary_rows: List[dict] = []
    judge_tasks: List[dict] = []
    selected_count = 0
    skipped_for_low_positive = 0
    seen = 0

    logger.info(
        "Exporting concept presence subset: model={} requested_images={} topk_concepts={} class_source={}",
        model_name,
        len(selected_positions),
        cli_args.topk_concepts,
        cli_args.class_source,
    )

    for images, concept_one_hot, targets in tqdm(loader, desc="Export concept presence subset"):
        concept_space, logits, _gt_transform = _get_batch_model_state(
            model_name=model_name,
            load_path=cli_args.load_path,
            args=ckpt_args,
            images=images,
        )
        concept_space = concept_space.detach().cpu()
        logits = logits.detach().cpu()
        preds = logits.argmax(dim=1)

        for b in range(images.shape[0]):
            subset_pos = seen + b
            if subset_pos not in selected_pos_set:
                continue

            dataset_index = int(subset_indices[subset_pos])
            target_idx = int(targets[b].item())
            pred_idx = int(preds[b].item())
            reference_class_idx = pred_idx if cli_args.class_source == "pred" else target_idx
            contribution = class_weights[reference_class_idx] * concept_space[b]
            activation_scores = concept_space[b]
            positive_mask = (contribution > 0) & (activation_scores > 0)
            positive_indices = positive_mask.nonzero(as_tuple=False).flatten().tolist()

            if len(positive_indices) < int(cli_args.min_positive_concepts):
                skipped_for_low_positive += 1
                continue

            positive_indices = sorted(
                positive_indices,
                key=lambda idx: float(contribution[idx].item()),
                reverse=True,
            )
            chosen_indices = positive_indices[: int(cli_args.topk_concepts)]

            image_pil = base_dataset.get_image_pil(dataset_index).convert("RGB")
            image_name = f"image_{selected_count:04d}_idx_{dataset_index}.jpg"
            image_rel = Path("images") / image_name
            image_pil.save(images_dir / image_name)

            gt_concept_mask = concept_one_hot[b].detach().cpu() > 0
            case_payload = {
                "subset_rank": int(selected_count),
                "subset_position": int(subset_pos),
                "dataset_index": dataset_index,
                "image_file": str(image_rel),
                "target_class_idx": target_idx,
                "target_class_name": class_names[target_idx],
                "pred_class_idx": pred_idx,
                "pred_class_name": class_names[pred_idx],
                "reference_class_idx": reference_class_idx,
                "reference_class_name": class_names[reference_class_idx],
                "class_source": cli_args.class_source,
                "model_name": model_name,
                "checkpoint_path": cli_args.load_path,
                "concepts": [],
            }

            for rank_idx, concept_idx in enumerate(chosen_indices):
                concept_name = concepts[concept_idx]
                stem = f"image_{selected_count:04d}_idx_{dataset_index}_concept_{rank_idx:02d}_{concept_idx:03d}_{_slugify(concept_name)}"
                concept_record = {
                    "rank_in_image": int(rank_idx),
                    "concept_index": int(concept_idx),
                    "concept_name": concept_name,
                    "annotation_present": bool(gt_concept_mask[concept_idx].item()),
                    "contribution_to_reference_class": float(contribution[concept_idx].item()),
                    "activation_score": float(activation_scores[concept_idx].item()),
                    "normalized_concept_score": float(concept_space[b, concept_idx].item()),
                }
                case_payload["concepts"].append(concept_record)
                judge_tasks.append(
                    {
                        "task_id": stem,
                        "task_type": "concept_presence",
                        "subset_rank": int(selected_count),
                        "dataset_index": dataset_index,
                        "model_name": model_name,
                        "checkpoint_path": cli_args.load_path,
                        "image_file": str(image_rel),
                        "concept_name": concept_name,
                        "metadata": {
                            "concept_index": int(concept_idx),
                            "annotation_present": bool(gt_concept_mask[concept_idx].item()),
                            "contribution_to_reference_class": float(contribution[concept_idx].item()),
                            "activation_score": float(activation_scores[concept_idx].item()),
                            "normalized_concept_score": float(concept_space[b, concept_idx].item()),
                        },
                        "prompt_template": JUDGE_PROMPT_TEMPLATE.format(
                            concept_name=concept_name,
                            model_name=model_name,
                        ),
                        "expected_response_schema": JUDGE_RESPONSE_SCHEMA,
                    }
                )

            case_path = cases_dir / f"image_{selected_count:04d}_idx_{dataset_index}.json"
            case_path.write_text(json.dumps(case_payload, indent=2))
            summary_rows.append(case_payload)
            selected_count += 1

        seen += images.shape[0]

    manifest = {
        "metadata": {
            "load_path": cli_args.load_path,
            "model_name": model_name,
            "dataset": ckpt_args.dataset,
            "annotation_dir": ckpt_args.annotation_dir,
            "selection": cli_args.selection,
            "seed": int(cli_args.seed),
            "requested_num_images": int(requested_num_images),
            "exported_num_images": int(selected_count),
            "topk_concepts": int(cli_args.topk_concepts),
            "class_source": cli_args.class_source,
            "min_positive_concepts": int(cli_args.min_positive_concepts),
            "skipped_for_low_positive_concepts": int(skipped_for_low_positive),
        },
        "images": summary_rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    with open(output_dir / "judge_tasks.jsonl", "w") as f:
        for row in judge_tasks:
            f.write(json.dumps(row) + "\n")

    logger.info(
        "Saved concept presence subset to {} with {} images and {} concept tasks",
        output_dir,
        selected_count,
        len(judge_tasks),
    )


if __name__ == "__main__":
    main()
