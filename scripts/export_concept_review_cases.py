import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

from data import utils as data_utils
from data.concept_dataset import _unwrap_dataset_indices
from scripts.evaluate_concept_ranking import (
    _build_test_loader,
    _lf_scores,
    _load_checkpoint_args,
    _load_concepts,
    _parse_ks,
    _resolve_num_images,
    _salf_scores,
    _savlg_scores,
    _vlg_scores,
)
from methods.common import load_run_info


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export review bundles with image, GT concepts, and top-k concept "
            "activations for manual inspection."
        )
    )
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--savlg_score_source",
        type=str,
        default="final",
        choices=["final", "global", "spatial"],
    )
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max_cases", type=int, default=10)
    parser.add_argument(
        "--selection",
        type=str,
        default="missed_at_k",
        choices=["first", "missed_at_k", "random"],
    )
    parser.add_argument(
        "--focus_k",
        type=int,
        default=10,
        help="Used with selection=missed_at_k.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def _score_batch(
    model_name: str,
    load_path: str,
    args,
    images: torch.Tensor,
    score_source: str,
) -> torch.Tensor:
    if model_name == "lf_cbm":
        return _lf_scores(load_path, args, images)
    if model_name == "vlg_cbm":
        return _vlg_scores(load_path, args, images)
    if model_name == "salf_cbm":
        return _salf_scores(load_path, args, images)
    if model_name == "savlg_cbm":
        return _savlg_scores(load_path, args, images, score_source)
    raise NotImplementedError(f"Unsupported model_name={model_name}")


def _compute_ranks(scores: torch.Tensor) -> torch.Tensor:
    batch_size, n_concepts = scores.shape
    order = scores.argsort(dim=1, descending=True)
    positions = (
        torch.arange(1, n_concepts + 1, device=scores.device, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, positions)
    return ranks


def _prepare_overlay(image_pil: Image.Image, annotations: List[dict], out_path: Path) -> None:
    fig = data_utils.plot_annotations(image_pil, annotations)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.clf()


def _get_case_annotations(base_dataset, idx: int) -> List[dict]:
    if hasattr(base_dataset, "_normalize_annotations") and hasattr(base_dataset, "_load_raw_data"):
        return base_dataset._normalize_annotations(base_dataset._load_raw_data(idx))
    if hasattr(base_dataset, "get_annotations"):
        return base_dataset.get_annotations(idx)
    raise AttributeError("Dataset does not expose an annotation access method.")


def _choose_candidates(
    sample_rows: List[dict],
    selection: str,
    focus_k: int,
    max_cases: int,
    seed: int,
) -> List[dict]:
    if selection == "first":
        return sample_rows[:max_cases]
    if selection == "random":
        rng = random.Random(seed)
        rows = sample_rows[:]
        rng.shuffle(rows)
        return rows[:max_cases]

    missed = [row for row in sample_rows if not row["hit_at_focus_k"]]
    if len(missed) >= max_cases:
        return missed[:max_cases]
    extras = [row for row in sample_rows if row["hit_at_focus_k"]]
    return (missed + extras)[:max_cases]


def main() -> None:
    args_ns = _parse_args()
    run_info = load_run_info(args_ns.load_path)
    args = _load_checkpoint_args(args_ns.load_path, args_ns.device, args_ns.annotation_dir)
    model_name = run_info.get("model_name", "vlg_cbm")
    concepts = _load_concepts(args_ns.load_path)
    num_images = _resolve_num_images(args_ns)
    focus_k = min(args_ns.focus_k, len(concepts))
    topk = min(args_ns.topk, len(concepts))
    _ = _parse_ks(str(focus_k), len(concepts))

    torch.manual_seed(args_ns.seed)
    np.random.seed(args_ns.seed)
    random.seed(args_ns.seed)

    test_loader = _build_test_loader(
        model_name,
        args_ns.load_path,
        args,
        concepts,
        args_ns.batch_size,
        args_ns.num_workers,
        num_images,
    )

    base_dataset, subset_indices = _unwrap_dataset_indices(test_loader.dataset)
    class_names = data_utils.get_classes(args.dataset)

    logger.info(
        "Exporting concept review cases: model={} images={} selection={} focus_k={} topk={}",
        model_name,
        len(test_loader.dataset),
        args_ns.selection,
        focus_k,
        topk,
    )

    sample_rows: List[dict] = []
    seen = 0
    for images, concept_one_hot, targets in tqdm(test_loader):
        scores = _score_batch(
            model_name,
            args_ns.load_path,
            args,
            images,
            args_ns.savlg_score_source,
        ).detach()
        ranks = _compute_ranks(scores)
        topk_vals, topk_idx = scores.topk(k=topk, dim=1)
        gt_mask = concept_one_hot > 0

        for b in range(images.shape[0]):
            subset_pos = seen + b
            base_idx = int(subset_indices[subset_pos])
            gt_indices = gt_mask[b].nonzero(as_tuple=False).flatten().tolist()
            if not gt_indices:
                continue

            gt_ranks = sorted(
                (
                    {
                        "concept": concepts[idx],
                        "rank": int(ranks[b, idx].item()),
                    }
                    for idx in gt_indices
                ),
                key=lambda item: item["rank"],
            )
            hit_at_focus_k = any(item["rank"] <= focus_k for item in gt_ranks)
            sample_rows.append(
                {
                    "subset_index": subset_pos,
                    "dataset_index": base_idx,
                    "target_idx": int(targets[b].item()),
                    "target_name": class_names[int(targets[b].item())],
                    "gt_concepts": [concepts[idx] for idx in gt_indices],
                    "gt_ranks": gt_ranks,
                    "hit_at_focus_k": hit_at_focus_k,
                    "topk_predictions": [
                        {
                            "concept": concepts[int(idx.item())],
                            "score": float(val.item()),
                        }
                        for idx, val in zip(topk_idx[b], topk_vals[b])
                    ],
                }
            )
        seen += images.shape[0]

    selected = _choose_candidates(
        sample_rows,
        args_ns.selection,
        focus_k,
        args_ns.max_cases,
        args_ns.seed,
    )

    output_dir = Path(args_ns.output_dir)
    images_dir = output_dir / "images"
    overlays_dir = output_dir / "gt_overlays"
    case_dir = output_dir / "cases"
    images_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    manifest_cases = []
    for case_idx, row in enumerate(selected):
        base_idx = int(row["dataset_index"])
        image_pil = base_dataset.get_image_pil(base_idx).convert("RGB")
        annotations = _get_case_annotations(base_dataset, base_idx)

        image_name = f"case_{case_idx:02d}_idx_{base_idx}.jpg"
        overlay_name = f"case_{case_idx:02d}_idx_{base_idx}_gt.png"
        image_out = images_dir / image_name
        overlay_out = overlays_dir / overlay_name

        image_pil.save(image_out)
        _prepare_overlay(image_pil, annotations, overlay_out)

        case_payload = dict(row)
        case_payload["image_file"] = str(Path("images") / image_name)
        case_payload["gt_overlay_file"] = str(Path("gt_overlays") / overlay_name)
        case_payload["annotations"] = annotations

        case_path = case_dir / f"case_{case_idx:02d}_idx_{base_idx}.json"
        case_path.write_text(json.dumps(case_payload, indent=2))
        manifest_cases.append(case_payload)

    summary = {
        "metadata": {
            "load_path": args_ns.load_path,
            "model_name": model_name,
            "dataset": args.dataset,
            "annotation_dir": args.annotation_dir,
            "score_source": args_ns.savlg_score_source if model_name == "savlg_cbm" else "default",
            "num_images_requested": num_images,
            "topk": topk,
            "focus_k": focus_k,
            "selection": args_ns.selection,
            "max_cases": args_ns.max_cases,
        },
        "num_candidate_cases": len(sample_rows),
        "num_exported_cases": len(manifest_cases),
        "cases": manifest_cases,
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
    logger.info("Saved {} review cases to {}", len(manifest_cases), output_dir)


if __name__ == "__main__":
    main()
