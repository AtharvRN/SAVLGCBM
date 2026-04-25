import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from tqdm import tqdm

from data import utils as data_utils
from data.concept_dataset import _unwrap_dataset_indices
from methods.common import load_run_info
from scripts.evaluate_concept_interventions import (
    _build_test_loader,
    _load_checkpoint_args,
    _load_concepts,
    _resolve_num_images,
    _salf_state,
    _savlg_state,
)
from scripts.visualize_savlg_examples import _normalize_map, _overlay_heatmap


JUDGE_PROMPT_TEMPLATE = """You are evaluating one concept-region pair for an interpretability study.

You will be shown:
1. the original image
2. a heatmap overlay highlighting the model's activated region for a single concept

Concept: {concept_name}
Model: {model_name}

Return a JSON object with exactly these fields:
- region_matches_concept: one of ["yes", "partial", "no", "unsure"]
- region_matches_concept_confidence: float in [0, 1]
- rationale_short: short string, max 30 words

Judging rules:
- region_matches_concept should answer whether the highlighted region actually corresponds to the named concept.
- Use "unsure" when the concept is too subtle, ambiguous, occluded, or the image resolution is insufficient.
- Do not infer from class label alone. Judge only from visible evidence in the image and overlay.
"""


JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "region_matches_concept",
        "region_matches_concept_confidence",
        "rationale_short",
    ],
    "properties": {
        "region_matches_concept": {
            "type": "string",
            "enum": ["yes", "partial", "no", "unsure"],
        },
        "region_matches_concept_confidence": {
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
            "Export spatial-faithfulness judge tasks for SALF/SAVLG checkpoints."
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
        "--savlg_score_source",
        type=str,
        default="final",
        choices=["final", "global", "spatial"],
    )
    parser.add_argument(
        "--map_normalization",
        type=str,
        default="concept_zscore_minmax",
        choices=["minmax", "sigmoid", "concept_zscore_minmax"],
    )
    parser.add_argument(
        "--min_positive_concepts",
        type=int,
        default=1,
    )
    return parser.parse_args()


def _normalize_map_with_mode(x: torch.Tensor, mode: str) -> torch.Tensor:
    x = x.detach().float()
    mode = str(mode).lower()
    if mode == "minmax":
        return _normalize_map(x)
    if mode == "sigmoid":
        return torch.sigmoid(x)
    if mode == "concept_zscore_minmax":
        x = (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)
        return _normalize_map(x)
    raise ValueError(f"Unsupported map_normalization={mode}")


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


def _ensure_spatial_model(model_name: str) -> None:
    if model_name not in {"salf_cbm", "savlg_cbm"}:
        raise NotImplementedError(
            f"export_spatial_judge_subset supports only salf_cbm/savlg_cbm, got {model_name}."
        )


def _build_batch_state(
    model_name: str,
    load_path: str,
    args,
    images: torch.Tensor,
):
    if model_name == "salf_cbm":
        backbone, concept_layer, mean, std, final_layer = _salf_state(load_path, args)
        with torch.no_grad():
            feats = backbone(images.to(args.device))
            native_maps = concept_layer(feats)
            activation_scores = F.adaptive_avg_pool2d(native_maps, 1).flatten(1)
            concept_space = (activation_scores - mean.to(args.device)) / std.to(args.device)
            logits = final_layer(concept_space)
        return {
            "native_maps": native_maps,
            "activation_scores": activation_scores,
            "concept_space": concept_space,
            "logits": logits,
            "final_layer": final_layer,
        }

    backbone, concept_layer, mean, std, final_layer = _savlg_state(load_path, args)
    from methods.savlg import (
        compute_savlg_concept_logits,
        forward_savlg_backbone,
        forward_savlg_concept_layer,
    )

    with torch.no_grad():
        feats = forward_savlg_backbone(backbone, images.to(args.device), args)
        global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
        global_logits, spatial_logits, final_logits = compute_savlg_concept_logits(
            global_outputs,
            spatial_maps,
            args,
        )
        if getattr(args, "savlg_score_source", "final") == "global":
            activation_scores = global_logits
        elif getattr(args, "savlg_score_source", "final") == "spatial":
            activation_scores = spatial_logits
        else:
            activation_scores = final_logits
        concept_space = (final_logits - mean.to(args.device)) / std.to(args.device)
        logits = final_layer(concept_space)
    return {
        "native_maps": spatial_maps,
        "activation_scores": activation_scores,
        "concept_space": concept_space,
        "logits": logits,
        "final_layer": final_layer,
        "global_logits": global_logits,
        "spatial_logits": spatial_logits,
        "final_logits": final_logits,
    }


def _make_dirs(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        "images": output_dir / "images",
        "overlays": output_dir / "overlays",
        "maps_native": output_dir / "maps_native",
        "maps_upsampled": output_dir / "maps_upsampled",
        "cases": output_dir / "cases",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_static_judge_files(output_dir: Path) -> None:
    (output_dir / "judge_prompt_template.txt").write_text(JUDGE_PROMPT_TEMPLATE)
    (output_dir / "judge_response_schema.json").write_text(json.dumps(JUDGE_RESPONSE_SCHEMA, indent=2))


def main() -> None:
    os.chdir(Path(__file__).resolve().parents[1])
    cli_args = _parse_args()
    run_info = load_run_info(cli_args.load_path)
    model_name = run_info.get("model_name", run_info["args"]["model_name"])
    _ensure_spatial_model(model_name)

    ckpt_args = _load_checkpoint_args(cli_args.load_path, cli_args.device, cli_args.annotation_dir)
    if model_name == "savlg_cbm":
        setattr(ckpt_args, "savlg_score_source", cli_args.savlg_score_source)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    dirs = _make_dirs(output_dir)
    _write_static_judge_files(output_dir)

    summary_rows: List[dict] = []
    judge_tasks: List[dict] = []
    selected_count = 0
    skipped_for_low_positive = 0
    seen = 0

    logger.info(
        "Exporting spatial judge subset: model={} requested_images={} topk_concepts={} class_source={} score_source={}",
        model_name,
        len(selected_positions),
        cli_args.topk_concepts,
        cli_args.class_source,
        cli_args.savlg_score_source if model_name == "savlg_cbm" else "pooled_map",
    )

    for images, concept_one_hot, targets in tqdm(loader, desc="Export spatial judge subset"):
        batch_state = _build_batch_state(model_name, cli_args.load_path, ckpt_args, images)
        native_maps = batch_state["native_maps"].detach().cpu()
        activation_scores = batch_state["activation_scores"].detach().cpu()
        concept_space = batch_state["concept_space"].detach().cpu()
        logits = batch_state["logits"].detach().cpu()
        class_weights = batch_state["final_layer"].weight.detach().cpu()

        global_logits = batch_state.get("global_logits")
        spatial_logits = batch_state.get("spatial_logits")
        final_logits = batch_state.get("final_logits")
        if global_logits is not None:
            global_logits = global_logits.detach().cpu()
        if spatial_logits is not None:
            spatial_logits = spatial_logits.detach().cpu()
        if final_logits is not None:
            final_logits = final_logits.detach().cpu()

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
            positive_mask = (contribution > 0) & (activation_scores[b] > 0)
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
            image_np = np.asarray(image_pil, dtype=np.float32) / 255.0
            img_w, img_h = image_pil.size

            image_name = f"image_{selected_count:04d}_idx_{dataset_index}.jpg"
            image_rel = Path("images") / image_name
            image_pil.save(dirs["images"] / image_name)

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
                native_map = native_maps[b, concept_idx]
                upsampled_map = F.interpolate(
                    native_map.unsqueeze(0).unsqueeze(0),
                    size=(img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                display_map = _normalize_map_with_mode(upsampled_map, cli_args.map_normalization)
                overlay_np = _overlay_heatmap(image_np, display_map)

                stem = f"image_{selected_count:04d}_idx_{dataset_index}_concept_{rank_idx:02d}_{concept_idx:03d}_{_slugify(concept_name)}"
                overlay_rel = Path("overlays") / f"{stem}.png"
                native_map_rel = Path("maps_native") / f"{stem}.npy"
                upsampled_map_rel = Path("maps_upsampled") / f"{stem}.npy"

                np.save(dirs["maps_native"] / f"{stem}.npy", native_map.numpy())
                np.save(dirs["maps_upsampled"] / f"{stem}.npy", display_map.numpy())

                overlay_img = (overlay_np * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(overlay_img).save(dirs["overlays"] / f"{stem}.png")

                concept_record = {
                    "rank_in_image": int(rank_idx),
                    "concept_index": int(concept_idx),
                    "concept_name": concept_name,
                    "annotation_present": bool(gt_concept_mask[concept_idx].item()),
                    "contribution_to_reference_class": float(contribution[concept_idx].item()),
                    "activation_score": float(activation_scores[b, concept_idx].item()),
                    "normalized_concept_score": float(concept_space[b, concept_idx].item()),
                    "overlay_file": str(overlay_rel),
                    "native_map_file": str(native_map_rel),
                    "upsampled_map_file": str(upsampled_map_rel),
                }
                if global_logits is not None and spatial_logits is not None and final_logits is not None:
                    concept_record["savlg_global_logit"] = float(global_logits[b, concept_idx].item())
                    concept_record["savlg_spatial_logit"] = float(spatial_logits[b, concept_idx].item())
                    concept_record["savlg_final_logit"] = float(final_logits[b, concept_idx].item())

                case_payload["concepts"].append(concept_record)
                judge_tasks.append(
                    {
                        "task_id": stem,
                        "task_type": "spatial_faithfulness",
                        "subset_rank": int(selected_count),
                        "dataset_index": dataset_index,
                        "model_name": model_name,
                        "checkpoint_path": cli_args.load_path,
                        "image_file": str(image_rel),
                        "overlay_file": str(overlay_rel),
                        "concept_name": concept_name,
                        "metadata": {
                            "concept_index": int(concept_idx),
                            "annotation_present": bool(gt_concept_mask[concept_idx].item()),
                            "contribution_to_reference_class": float(contribution[concept_idx].item()),
                            "activation_score": float(activation_scores[b, concept_idx].item()),
                            "normalized_concept_score": float(concept_space[b, concept_idx].item()),
                        },
                        "prompt_template": JUDGE_PROMPT_TEMPLATE.format(
                            concept_name=concept_name,
                            model_name=model_name,
                        ),
                        "expected_response_schema": JUDGE_RESPONSE_SCHEMA,
                    }
                )

            case_path = dirs["cases"] / f"image_{selected_count:04d}_idx_{dataset_index}.json"
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
            "map_normalization": cli_args.map_normalization,
            "savlg_score_source": cli_args.savlg_score_source if model_name == "savlg_cbm" else "pooled_map",
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
        "Saved spatial judge subset to {} with {} images and {} concept tasks",
        output_dir,
        selected_count,
        len(judge_tasks),
    )


if __name__ == "__main__":
    main()
