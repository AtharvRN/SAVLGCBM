#!/usr/bin/env python3
"""Generate or plan SAM3 concept pseudo-mask caches.

This first-pass scaffold is intentionally subset-friendly and does not use CUB
concept-to-part mappings. In dry-run mode it writes the manifest and per-record
metadata only; the real SAM3 backend can be added behind `Sam3ConceptMaskRunner`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.sam3_concept_mask_cache import (
    CACHE_SCHEMA_VERSION,
    Sam3CacheLayout,
    build_manifest,
    build_manifest_record,
    load_concept_set,
    relative_record_paths,
    resolve_image_size,
    resolve_sample_path,
    select_indices,
    write_json,
)


class Sam3ConceptMaskRunner:
    """Thin backend boundary for image+text to mask inference."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = str(config.get("backend", "medsam3_lora")).lower()
        if self.backend in {"sam3", "medsam3", "medsam3_lora"}:
            self._init_medsam3_lora(config)
        else:
            raise ValueError(f"Unsupported SAM3 backend: {self.backend}")

    def _init_medsam3_lora(self, config: Dict[str, Any]) -> None:
        repo_path = Path(config.get("repo_path", "/workspace/MedSAM3")).expanduser()
        if not repo_path.is_dir():
            raise FileNotFoundError(f"MedSAM3 repo_path does not exist: {repo_path}")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        model_config = str(
            config.get("model_config")
            or config.get("config_path")
            or repo_path / "configs" / "full_lora_config.yaml"
        )
        weights = str(config.get("lora_weights") or config.get("checkpoint") or "")
        if not weights:
            weights = None
        resolution = int(config.get("resolution", 512))
        score_threshold = float(config.get("score_threshold", 0.5))
        nms_iou = float(config.get("nms_iou_threshold", 0.5))
        device = str(config.get("device", "cuda"))

        cwd = os.getcwd()
        try:
            os.chdir(repo_path)
            from infer_sam import SAM3LoRAInference

            self.inferencer = SAM3LoRAInference(
                config_path=model_config,
                weights_path=weights,
                resolution=resolution,
                detection_threshold=score_threshold,
                nms_iou_threshold=nms_iou,
                device=device,
            )
        finally:
            os.chdir(cwd)

    def predict(self, image_path: str, concept: str) -> Dict[str, Any]:
        if self.backend in {"sam3", "medsam3", "medsam3_lora"}:
            results = self.inferencer.predict(image_path, [concept])
            result = results.get(0, {})
            scores = result.get("scores")
            masks = result.get("masks")
            boxes = result.get("boxes")
            if masks is None or scores is None or int(result.get("num_detections", 0)) <= 0:
                return {
                    "status": "no_mask",
                    "mask": None,
                    "score": None,
                    "bbox_xyxy": None,
                    "num_detections": int(result.get("num_detections", 0)),
                }
            scores_np = np.asarray(scores, dtype=np.float32)
            masks_np = np.asarray(masks)
            boxes_np = None if boxes is None else np.asarray(boxes, dtype=np.float32)
            best_idx = int(scores_np.argmax())
            mask = masks_np[best_idx].astype(bool)
            bbox = None
            if boxes_np is not None and best_idx < len(boxes_np):
                bbox = [float(x) for x in boxes_np[best_idx].tolist()]
            return {
                "status": "ok",
                "mask": mask,
                "score": float(scores_np[best_idx]),
                "bbox_xyxy": bbox,
                "num_detections": int(len(scores_np)),
            }
        raise AssertionError(f"Unhandled backend: {self.backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SAM3 concept pseudo-mask cache scaffolds")
    parser.add_argument("--config", required=True, help="Path to SAM3 concept-mask config JSON")
    parser.add_argument("--split", default=None, choices=["train", "val", "test"], help="Override config split")
    parser.add_argument("--dry_run", action="store_true", help="Write manifest/metadata without running SAM3")
    parser.add_argument("--run", action="store_true", help="Run SAM3 inference even if config dry_run is true")
    parser.add_argument("--max_images", type=int, default=None, help="Override subset.max_images")
    parser.add_argument("--max_concepts", type=int, default=None, help="Override subset.max_concepts")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing manifest")
    return parser.parse_args()


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _split_to_dataset_name(dataset: str, split: str) -> str:
    if split == "train":
        return f"{dataset}_train"
    if split in {"val", "test"}:
        return f"{dataset}_val"
    raise ValueError(f"Unsupported split: {split}")


def _load_image_dataset(dataset: str, split: str):
    if dataset != "cub":
        raise ValueError(
            "This scaffold currently supports dataset='cub'. Add other datasets here "
            "once their file-backed image roots are needed."
        )
    from torchvision import datasets

    dataset_folder = Path(os.environ.get("DATASET_FOLDER", "datasets"))
    split_dir = "train" if split == "train" else "test"
    root = dataset_folder / "CUB" / split_dir
    if not root.is_dir():
        raise FileNotFoundError(
            f"CUB split directory not found: {root}. Set DATASET_FOLDER or create the CUB dataset first."
        )
    return datasets.ImageFolder(str(root), transform=None)


def _concept_indices(total: int, subset_cfg: Dict[str, Any], max_concepts: int | None) -> List[int]:
    configured = subset_cfg.get("concept_indices")
    selected = select_indices(total, explicit_indices=configured, max_items=max_concepts)
    concept_names = subset_cfg.get("concepts")
    if concept_names:
        raise ValueError(
            "subset.concepts is reserved for a later name-based selector. Use concept_indices "
            "or max_concepts in this scaffold."
        )
    return selected


def _record_metadata(record: Dict[str, Any], dry_run: bool, sam3_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "mask_format": record["mask_format"],
        "record": record,
        "dry_run": bool(dry_run),
        "sam3": sam3_cfg,
        "expected_npz_payload": {
            "mask": "bool or uint8 array with shape [image_h, image_w] in original image coordinates",
            "score": "float confidence or null",
            "bbox_xyxy": "optional [x1, y1, x2, y2] in pixel coordinates",
        },
    }


def _write_mask_npz(path: Path, prediction: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = prediction["mask"].astype(np.uint8)
    score = np.asarray(prediction.get("score", np.nan), dtype=np.float32)
    bbox = prediction.get("bbox_xyxy")
    bbox_arr = np.asarray([] if bbox is None else bbox, dtype=np.float32)
    np.savez_compressed(path, mask=mask, score=score, bbox_xyxy=bbox_arr)


def _write_preview(image_path: str, mask: np.ndarray, output_path: Path, alpha: float = 0.45) -> None:
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image).copy()
    if mask.shape[:2] != image_np.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape[:2]} does not match image shape {image_np.shape[:2]} for {image_path}"
        )
    color = np.array([255, 80, 32], dtype=np.float32)
    image_np[mask] = ((1.0 - alpha) * image_np[mask].astype(np.float32) + alpha * color).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_np).save(output_path)


def build_records(
    dataset_obj,
    split: str,
    image_indices: Sequence[int],
    concepts: Sequence[str],
    concept_indices: Sequence[int],
    backend: str,
    prompt_template: str,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    status = "planned" if dry_run else "pending"
    for dataset_index in tqdm(image_indices, desc=f"SAM3 cache plan ({split}) images"):
        image_path = resolve_sample_path(dataset_obj, dataset_index)
        image_size = resolve_image_size(dataset_obj, dataset_index)
        for concept_index in concept_indices:
            concept = concepts[int(concept_index)]
            records.append(
                build_manifest_record(
                    split=split,
                    dataset_index=int(dataset_index),
                    image_path=image_path,
                    image_size=image_size,
                    concept_index=int(concept_index),
                    concept=concept,
                    backend=backend,
                    prompt_template=prompt_template,
                    status=status,
                )
            )
    return records


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)

    dataset_name = str(config.get("dataset", "cub"))
    split = str(args.split or config.get("split", "train"))
    subset_cfg = dict(config.get("subset", {}))
    if args.max_images is not None:
        subset_cfg["max_images"] = args.max_images
    if args.max_concepts is not None:
        subset_cfg["max_concepts"] = args.max_concepts

    concept_set = str(config["concept_set"])
    concepts = load_concept_set(concept_set)
    concept_indices = _concept_indices(
        len(concepts),
        subset_cfg,
        subset_cfg.get("max_concepts"),
    )
    _ = _split_to_dataset_name(dataset_name, split)
    dataset_obj = _load_image_dataset(dataset_name, split)
    image_indices = select_indices(
        len(dataset_obj),
        explicit_indices=subset_cfg.get("image_indices"),
        max_items=subset_cfg.get("max_images"),
    )

    sam3_cfg = dict(config.get("sam3", {}))
    backend = str(sam3_cfg.get("backend", "sam3"))
    prompt_template = str(sam3_cfg.get("prompt_template", "{concept}"))
    dry_run = False if args.run else bool(args.dry_run or config.get("dry_run", True))

    output_root = Path(config.get("output_root", "saved_activations/sam3_concept_masks"))
    cache_name = str(config.get("cache_name", f"{dataset_name}_concept_masks_v1"))
    layout = Sam3CacheLayout(root=output_root / cache_name, split=split)
    if layout.manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest already exists: {layout.manifest_path}. Use --overwrite to replace it.")
    layout.ensure_dirs()

    records = build_records(
        dataset_obj=dataset_obj,
        split=split,
        image_indices=image_indices,
        concepts=concepts,
        concept_indices=concept_indices,
        backend=backend,
        prompt_template=prompt_template,
        dry_run=dry_run,
    )

    cache_cfg = dict(config.get("cache", {}))
    if not dry_run:
        continue_on_error = bool(sam3_cfg.get("continue_on_error", True))
        try:
            runner = Sam3ConceptMaskRunner(sam3_cfg)
        except Exception as exc:
            if not continue_on_error:
                raise
            failure = f"backend_init:{type(exc).__name__}: {exc}"
            for record in records:
                record["status"] = "error"
                record["failure_reason"] = failure
        else:
            for record in tqdm(records, desc=f"SAM3 inference ({split})"):
                image_path = record.get("image_path")
                if image_path is None:
                    raise RuntimeError("Real SAM3 generation currently requires file-backed datasets.")
                try:
                    prediction = runner.predict(str(image_path), str(record["concept"]))
                    record["status"] = prediction["status"]
                    record["score"] = prediction.get("score")
                    record["num_detections"] = prediction.get("num_detections")
                    record["bbox_xyxy"] = prediction.get("bbox_xyxy")
                    record["failure_reason"] = None if prediction["status"] == "ok" else prediction["status"]
                    if prediction["status"] == "ok":
                        mask_path = layout.root / split / str(record["mask_path"])
                        _write_mask_npz(mask_path, prediction)
                        if bool(cache_cfg.get("write_preview_png", True)):
                            preview_path = layout.root / split / str(record["preview_path"])
                            _write_preview(
                                str(image_path),
                                prediction["mask"],
                                preview_path,
                                alpha=float(cache_cfg.get("preview_alpha", 0.45)),
                            )
                except Exception as exc:
                    record["status"] = "error"
                    record["failure_reason"] = f"{type(exc).__name__}: {exc}"
                    if not continue_on_error:
                        raise

    for record in records:
        rel_paths = relative_record_paths(record["dataset_index"], record["concept_index"])
        metadata_path = layout.root / split / rel_paths["metadata_path"]
        write_json(metadata_path, _record_metadata(record, dry_run=dry_run, sam3_cfg=sam3_cfg))

    manifest = build_manifest(
        dataset=dataset_name,
        split=split,
        concept_set=concept_set,
        concepts=concepts,
        records=records,
        config={
            "config_path": str(Path(args.config).resolve()),
            "dry_run": dry_run,
            "subset": subset_cfg,
            "sam3": sam3_cfg,
            "output_root": str(output_root),
            "cache_name": cache_name,
        },
    )
    write_json(layout.manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(layout.manifest_path),
                "images": len(image_indices),
                "concepts": len(concept_indices),
                "records": len(records),
                "dry_run": dry_run,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
