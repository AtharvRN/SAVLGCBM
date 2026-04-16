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
import inspect
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
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
    relative_candidate_preview_path,
    relative_record_paths,
    resolve_image_size,
    resolve_sample_path,
    select_indices,
    write_json,
)
from data.concept_grounding import build_grounding_specs


class Sam3ConceptMaskRunner:
    """Thin backend boundary for image+text to mask inference."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = str(config.get("backend", "sam3")).lower()
        self.mask_selection = str(config.get("mask_selection", "top_score")).lower()
        self.selection_score_power = float(config.get("selection_score_power", 1.0))
        self.selection_area_power = float(config.get("selection_area_power", 0.15))
        self.min_mask_area_ratio = (
            None if config.get("min_mask_area_ratio") is None else float(config.get("min_mask_area_ratio"))
        )
        self.max_mask_area_ratio = (
            None if config.get("max_mask_area_ratio") is None else float(config.get("max_mask_area_ratio"))
        )
        self.selection_fallback = str(config.get("selection_fallback", "top_score")).lower()
        self.candidate_top_k = max(
            1,
            int(config.get("candidate_top_k", config.get("max_masks_per_concept", 1))),
        )
        if self.backend == "sam3":
            self._init_base_sam3(config)
        elif self.backend == "groundingdino_sam3":
            self._init_groundingdino_sam3(config)
        else:
            raise ValueError(f"Unsupported SAM3 backend: {self.backend}")

    def _init_base_sam3(self, config: Dict[str, Any]) -> None:
        repo_path_raw = str(
            config.get("repo_path") or os.environ.get("SAM3_REPO_PATH", "/workspace/sam3")
        ).strip()
        if repo_path_raw:
            repo_path = Path(repo_path_raw).expanduser()
            if repo_path.exists() and str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))

        checkpoint_path = str(
            config.get("checkpoint_path")
            or config.get("checkpoint")
            or os.environ.get("SAM3_CHECKPOINT_PATH", "")
        ).strip()
        load_from_hf = bool(config.get("load_from_hf", not checkpoint_path))

        device = str(config.get("device", "cuda"))
        resolution = int(config.get("resolution", 1024))
        score_threshold = float(config.get("score_threshold", 0.5))
        nms_iou = float(config.get("nms_iou_threshold", 0.5))
        inference_dtype_raw = str(
            config.get(
                "inference_dtype",
                "bfloat16" if device.startswith("cuda") else "float32",
            )
        ).strip().lower()
        try:
            import sam3 as sam3_pkg
            from sam3.model_builder import build_sam3_image_model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "No module named 'sam3'. Install the SAM3 package via requirements.txt "
                "or clone it locally and set repo_path/SAM3_REPO_PATH."
            ) from exc

        sam3_package_dir = Path(sam3_pkg.__file__).resolve().parent
        bpe_path = str(
            config.get("bpe_path")
            or sam3_package_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        )

        build_kwargs = {
            "device": device,
            "compile": False,
            "eval_mode": True,
        }
        sig = inspect.signature(build_sam3_image_model)
        if "bpe_path" in sig.parameters:
            build_kwargs["bpe_path"] = bpe_path
        if "load_from_HF" in sig.parameters:
            build_kwargs["load_from_HF"] = load_from_hf
        if checkpoint_path and "checkpoint_path" in sig.parameters:
            build_kwargs["checkpoint_path"] = checkpoint_path
        elif checkpoint_path and "checkpoint" in sig.parameters:
            build_kwargs["checkpoint"] = checkpoint_path
        elif checkpoint_path and "weights_path" in sig.parameters:
            build_kwargs["weights_path"] = checkpoint_path
        elif checkpoint_path and "model_path" in sig.parameters:
            build_kwargs["model_path"] = checkpoint_path

        self.base_model = build_sam3_image_model(**build_kwargs)
        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
        }
        if inference_dtype_raw not in dtype_map:
            raise ValueError(f"Unsupported inference_dtype: {inference_dtype_raw}")
        self._inference_dtype = dtype_map[inference_dtype_raw]
        self.base_model.to(device=device, dtype=self._inference_dtype)
        self.base_model.eval()
        self._use_autocast = device.startswith("cuda") and self._inference_dtype in {torch.bfloat16, torch.float16}
        self._install_module_input_dtype_hooks()

        self.base_transform = None
        try:
            from sam3.train.transforms.basic_for_api import (
                ComposeAPI,
                NormalizeAPI,
                RandomResizeAPI,
                ToTensorAPI,
            )

            self.base_transform = ComposeAPI(
                transforms=[
                    RandomResizeAPI(
                        sizes=resolution,
                        max_size=resolution,
                        square=True,
                        consistent_transform=False,
                    ),
                    ToTensorAPI(),
                    NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            self.base_batch_helpers = True
        except Exception as exc:
            raise ImportError(
                "Base SAM3 import succeeded, but the SAM3 API transform stack could not be loaded. "
                f"Missing or incompatible SAM3 package: {exc}"
            ) from exc

        from torchvision.ops import nms

        self._nms = nms
        self._device = device
        self._base_score_threshold = score_threshold
        self._base_nms_iou = nms_iou
        self._base_resolution = resolution

    def _load_sam3_image_predictor(self, config: Dict[str, Any]) -> None:
        repo_path_raw = str(
            config.get("repo_path") or os.environ.get("SAM3_REPO_PATH", "/workspace/sam3")
        ).strip()
        if repo_path_raw:
            repo_path = Path(repo_path_raw).expanduser()
            if repo_path.exists() and str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
        checkpoint_path = str(
            config.get("checkpoint_path")
            or config.get("checkpoint")
            or os.environ.get("SAM3_CHECKPOINT_PATH", "")
        ).strip()
        load_from_hf = bool(config.get("load_from_hf", not checkpoint_path))
        device = str(config.get("device", "cuda"))
        inference_dtype_raw = str(config.get("inference_dtype", "float32")).strip().lower()
        try:
            import sam3 as sam3_pkg
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "No module named 'sam3'. Install the SAM3 package via requirements.txt "
                "or clone it locally and set repo_path/SAM3_REPO_PATH."
            ) from exc
        sam3_package_dir = Path(sam3_pkg.__file__).resolve().parent
        bpe_path = str(
            config.get("bpe_path")
            or sam3_package_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        )
        build_kwargs = {"device": device, "compile": False, "eval_mode": True}
        sig = inspect.signature(build_sam3_image_model)
        if "bpe_path" in sig.parameters:
            build_kwargs["bpe_path"] = bpe_path
        if "load_from_HF" in sig.parameters:
            build_kwargs["load_from_HF"] = load_from_hf
        if checkpoint_path and "checkpoint_path" in sig.parameters:
            build_kwargs["checkpoint_path"] = checkpoint_path
        elif checkpoint_path and "checkpoint" in sig.parameters:
            build_kwargs["checkpoint"] = checkpoint_path
        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
        }
        if inference_dtype_raw not in dtype_map:
            raise ValueError(f"Unsupported inference_dtype: {inference_dtype_raw}")
        self._inference_dtype = dtype_map[inference_dtype_raw]
        self.base_model = build_sam3_image_model(**build_kwargs)
        self.base_model.to(device=device, dtype=self._inference_dtype)
        self.base_model.eval()
        self._use_autocast = device.startswith("cuda") and self._inference_dtype in {torch.bfloat16, torch.float16}
        self._install_module_input_dtype_hooks()
        self._device = device
        self._sam3_predictor = SAM3InteractiveImagePredictor(self.base_model)
        self._cached_image_path: Optional[str] = None

    def _init_groundingdino_sam3(self, config: Dict[str, Any]) -> None:
        from data.groundingdino_inference import GroundingDinoBoxRunner

        groundingdino_cfg = dict(config.get("groundingdino", {}))
        sam3_box_cfg = dict(config.get("sam3_box", {}))
        self._groundingdino = GroundingDinoBoxRunner(groundingdino_cfg)
        self._groundingdino_top_k = int(groundingdino_cfg.get("top_k", self.candidate_top_k))
        self._sam3_box_multimask_output = bool(sam3_box_cfg.get("multimask_output", True))
        self._sam3_box_mask_threshold = float(sam3_box_cfg.get("mask_threshold", 0.0))
        self._load_sam3_image_predictor({**config, **sam3_box_cfg})

    def _cast_batch_floats(self, value: Any) -> Any:
        if torch.is_tensor(value):
            if value.is_floating_point():
                return value.to(dtype=self._inference_dtype)
            return value
        if isinstance(value, dict):
            return {k: self._cast_batch_floats(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._cast_batch_floats(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._cast_batch_floats(v) for v in value)
        if hasattr(value, "__dict__"):
            for key, child in vars(value).items():
                setattr(value, key, self._cast_batch_floats(child))
            return value
        return value

    def _cast_nested_to_dtype(self, value: Any, dtype: torch.dtype) -> Any:
        if torch.is_tensor(value):
            if value.is_floating_point() and value.dtype != dtype:
                return value.to(dtype=dtype)
            return value
        if isinstance(value, dict):
            return {k: self._cast_nested_to_dtype(v, dtype) for k, v in value.items()}
        if isinstance(value, list):
            return [self._cast_nested_to_dtype(v, dtype) for v in value]
        if isinstance(value, tuple):
            return tuple(self._cast_nested_to_dtype(v, dtype) for v in value)
        return value

    def _module_input_dtype_pre_hook(self, module: torch.nn.Module, args: Tuple[Any, ...]) -> Tuple[Any, ...]:
        weight = getattr(module, "weight", None)
        if weight is None or not torch.is_tensor(weight) or not weight.is_floating_point():
            return args
        return tuple(self._cast_nested_to_dtype(arg, weight.dtype) for arg in args)

    def _install_module_input_dtype_hooks(self) -> None:
        supported = (
            torch.nn.Linear,
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.ConvTranspose1d,
            torch.nn.ConvTranspose2d,
            torch.nn.ConvTranspose3d,
        )
        for module in self.base_model.modules():
            if isinstance(module, supported):
                module.register_forward_pre_hook(self._module_input_dtype_pre_hook)

    def _candidate_bbox(
        self,
        boxes_np: Optional[np.ndarray],
        candidate_index: int,
        mask: np.ndarray,
    ) -> Optional[List[float]]:
        if boxes_np is not None and candidate_index < len(boxes_np):
            return [float(x) for x in boxes_np[candidate_index].tolist()]
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]

    def _selection_metric(self, score: float, area_ratio: float, mode: str) -> float:
        in_area_band = True
        if self.min_mask_area_ratio is not None and area_ratio < self.min_mask_area_ratio:
            in_area_band = False
        if self.max_mask_area_ratio is not None and area_ratio > self.max_mask_area_ratio:
            in_area_band = False

        if mode == "top_score":
            return float(score)
        if mode in {"score_area_constrained", "score_with_area_constraints"}:
            if not in_area_band:
                return float("-inf")
            return float((max(score, 1e-8) ** self.selection_score_power) * (max(area_ratio, 1e-8) ** self.selection_area_power))
        if mode in {"largest_valid", "largest_with_area_constraints"}:
            if not in_area_band:
                return float("-inf")
            return float(area_ratio)
        raise ValueError(f"Unsupported mask_selection mode: {mode}")

    def _select_candidate(
        self,
        candidates: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[int], str, Optional[str]]:
        ranked = []
        for idx, candidate in enumerate(candidates):
            metric = self._selection_metric(
                score=float(candidate["score"]),
                area_ratio=float(candidate["area_ratio"]),
                mode=self.mask_selection,
            )
            candidate["selection_metric"] = metric
            ranked.append((metric, float(candidate["score"]), -idx, idx))

        ranked.sort(reverse=True)
        if ranked and np.isfinite(ranked[0][0]):
            chosen_mode = self.mask_selection
            fallback = None
            return candidates[ranked[0][3]], [item[3] for item in ranked], chosen_mode, fallback

        if self.selection_fallback == "top_score":
            fallback_ranked = []
            for idx, candidate in enumerate(candidates):
                metric = self._selection_metric(
                    score=float(candidate["score"]),
                    area_ratio=float(candidate["area_ratio"]),
                    mode="top_score",
                )
                candidate["selection_metric"] = metric
                fallback_ranked.append((metric, -idx, idx))
            fallback_ranked.sort(reverse=True)
            return (
                candidates[fallback_ranked[0][2]],
                [item[2] for item in fallback_ranked],
                "top_score",
                self.mask_selection,
            )

        raise RuntimeError("No valid candidate remained after mask selection and no supported fallback was configured.")

    def predict(self, image_path: str, concept: str) -> Dict[str, Any]:
        if self.backend == "sam3":
            return self._predict_base_sam3(image_path, concept)
        if self.backend == "groundingdino_sam3":
            return self._predict_groundingdino_sam3(image_path, concept)
        raise AssertionError(f"Unhandled SAM3 backend: {self.backend}")

    def _select_sam3_box_mask(
        self,
        masks: np.ndarray,
        ious: np.ndarray,
        box_xyxy: Sequence[float],
        image_hw: Tuple[int, int],
        grounding_score: float,
    ) -> Dict[str, Any]:
        h, w = image_hw
        candidates = []
        for idx in range(int(masks.shape[0])):
            mask = masks[idx] > self._sam3_box_mask_threshold
            area_ratio = float(mask.mean())
            candidate = {
                "mask": mask.astype(bool),
                "score": float(ious[idx]),
                "bbox_xyxy": [float(x) for x in box_xyxy],
                "source_index": int(idx),
                "area_ratio": area_ratio,
                "grounding_score": float(grounding_score),
            }
            candidate["selection_metric"] = self._selection_metric(
                score=float(candidate["score"]),
                area_ratio=area_ratio,
                mode=self.mask_selection,
            )
            candidates.append(candidate)
        selected, ranked_indices, selected_mode, fallback_from = self._select_candidate(candidates)
        candidate_summaries = []
        candidate_payloads = []
        for rank, cand_idx in enumerate(ranked_indices[: self.candidate_top_k]):
            candidate = candidates[cand_idx]
            candidate_summaries.append(
                {
                    "candidate_rank": int(rank),
                    "source_index": int(candidate["source_index"]),
                    "score": float(candidate["score"]),
                    "grounding_score": float(candidate["grounding_score"]),
                    "area_ratio": float(candidate["area_ratio"]),
                    "bbox_xyxy": [float(x) for x in candidate["bbox_xyxy"]],
                    "selection_metric": float(candidate["selection_metric"]),
                    "selected": bool(candidate is selected),
                }
            )
            candidate_payloads.append({"candidate_rank": int(rank), "mask": candidate["mask"]})
        return {
            "status": "ok",
            "mask": selected["mask"],
            "score": float(selected["score"]),
            "bbox_xyxy": [float(x) for x in selected["bbox_xyxy"]],
            "area_ratio": float(selected["area_ratio"]),
            "grounding_score": float(grounding_score),
            "mask_selection": selected_mode,
            "selection_fallback_from": fallback_from,
            "selected_source_index": int(selected["source_index"]),
            "selected_selection_metric": float(selected["selection_metric"]),
            "candidate_count": int(len(candidates)),
            "candidate_summaries": candidate_summaries,
            "candidate_preview_payloads": candidate_payloads,
            "num_detections": int(len(candidates)),
        }

    def _predict_groundingdino_sam3(self, image_path: str, concept: str) -> Dict[str, Any]:
        boxes = self._groundingdino.predict_boxes(
            image_path=image_path,
            prompt=concept,
            top_k=self._groundingdino_top_k,
        )
        if not boxes:
            return {
                "status": "no_mask",
                "mask": None,
                "score": None,
                "bbox_xyxy": None,
                "num_detections": 0,
            }
        pil_image = Image.open(image_path).convert("RGB")
        if self._cached_image_path != image_path:
            self._sam3_predictor.set_image(pil_image)
            self._cached_image_path = image_path
        image_hw = (pil_image.size[1], pil_image.size[0])
        box_xyxy = np.asarray(boxes[0]["box_xyxy"], dtype=np.float32)
        masks, ious, _ = self._sam3_predictor.predict(
            box=box_xyxy,
            multimask_output=self._sam3_box_multimask_output,
            return_logits=False,
        )
        masks = np.asarray(masks)
        ious = np.asarray(ious)
        if masks.ndim == 2:
            masks = masks[None, ...]
        return self._select_sam3_box_mask(
            masks=masks,
            ious=ious,
            box_xyxy=box_xyxy,
            image_hw=image_hw,
            grounding_score=float(boxes[0]["score"]),
        )

    def _predict_base_sam3(self, image_path: str, concept: str) -> Dict[str, Any]:
        from PIL import Image as PILImage
        from sam3.train.data.collator import collate_fn_api
        from sam3.train.data.sam3_image_dataset import (
            Datapoint,
            FindQueryLoaded,
            Image as SAMImage,
            InferenceMetadata,
        )
        from sam3.model.utils.misc import copy_data_to_device

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        pil_image = PILImage.open(image_path).convert("RGB")
        w, h = pil_image.size
        sam_image = SAMImage(data=pil_image, objects=[], size=[h, w])
        query = FindQueryLoaded(
            query_text=concept,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=0,
                original_image_id=0,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            ),
        )
        datapoint = Datapoint(find_queries=[query], images=[sam_image])
        datapoint = self.base_transform(datapoint)
        batch = collate_fn_api([datapoint], dict_key="input")["input"]
        batch = copy_data_to_device(batch, self.base_model.device if hasattr(self.base_model, "device") else self._device, non_blocking=True)
        batch = self._cast_batch_floats(batch)
        with torch.inference_mode():
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=self._inference_dtype)
                if self._use_autocast
                else nullcontext()
            )
            with autocast_ctx:
                outputs = self.base_model(batch)
            last_output = outputs[-1]
            pred_logits = last_output["pred_logits"]
            pred_boxes = last_output["pred_boxes"]
            pred_masks = last_output.get("pred_masks", None)

        out_probs = pred_logits.sigmoid()
        scores = out_probs[0, :, :].max(dim=-1)[0]
        keep = scores > self._base_score_threshold
        num_keep = int(keep.sum().item())
        if num_keep <= 0:
            return {
                "status": "no_mask",
                "mask": None,
                "score": None,
                "bbox_xyxy": None,
                "num_detections": 0,
            }

        boxes_cxcywh = pred_boxes[0, keep]
        kept_scores = scores[keep]
        cx, cy, w_box, h_box = boxes_cxcywh.unbind(-1)
        x1 = (cx - w_box / 2) * w
        y1 = (cy - h_box / 2) * h
        x2 = (cx + w_box / 2) * w
        y2 = (cy + h_box / 2) * h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        keep_nms = self._nms(boxes_xyxy, kept_scores, self._base_nms_iou)
        boxes_xyxy = boxes_xyxy[keep_nms]
        kept_scores = kept_scores[keep_nms]
        if pred_masks is not None:
            masks_small = pred_masks[0, keep][keep_nms].sigmoid() > 0.5
            import torch.nn.functional as F

            masks_resized = F.interpolate(
                masks_small.unsqueeze(0).float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0) > 0.5
            masks_np = masks_resized.cpu().numpy()
        else:
            masks_np = None
        if masks_np is None:
            return {
                "status": "no_mask",
                "mask": None,
                "score": None,
                "bbox_xyxy": None,
                "num_detections": int(len(kept_scores)),
            }
        best_idx = int(kept_scores.argmax().item())
        return {
            "status": "ok",
            "mask": masks_np[best_idx].astype(bool),
            "score": float(kept_scores[best_idx].item()),
            "bbox_xyxy": [float(x) for x in boxes_xyxy[best_idx].tolist()],
            "num_detections": int(len(kept_scores)),
        }


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
        "candidate_summary_schema": {
            "candidate_rank": "rank after selection sorting, starting at 0",
            "source_index": "candidate index from SAM3 output after thresholding+NMS",
            "score": "SAM3 confidence score",
            "area_ratio": "mask area divided by image area",
            "bbox_xyxy": "candidate box in original image pixels",
            "selection_metric": "scalar used to rank candidates",
            "selected": "whether this candidate became the cached pseudo-mask",
            "candidate_preview_path": "optional overlay preview path for audit",
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


def _attach_candidate_preview_paths(
    record: Dict[str, Any],
    prediction: Dict[str, Any],
    layout: Sam3CacheLayout,
    split: str,
    image_path: str,
    preview_alpha: float,
    write_candidate_previews: bool,
) -> None:
    candidate_summaries = list(prediction.get("candidate_summaries", []))
    payloads = list(prediction.get("candidate_preview_payloads", []))
    payload_by_rank = {int(item["candidate_rank"]): item for item in payloads}
    for summary in candidate_summaries:
        candidate_rank = int(summary["candidate_rank"])
        rel_path = relative_candidate_preview_path(
            dataset_index=int(record["dataset_index"]),
            concept_index=int(record["concept_index"]),
            candidate_rank=candidate_rank,
        )
        summary["candidate_preview_path"] = rel_path
        payload = payload_by_rank.get(candidate_rank)
        if write_candidate_previews and payload is not None:
            _write_preview(
                image_path,
                payload["mask"],
                layout.root / split / rel_path,
                alpha=preview_alpha,
            )
    record["candidate_summaries"] = candidate_summaries


def build_records(
    dataset_obj,
    split: str,
    image_indices: Sequence[int],
    concepts: Sequence[str],
    concept_specs,
    backend: str,
    prompt_template: str,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    status = "planned" if dry_run else "pending"
    for dataset_index in tqdm(image_indices, desc=f"SAM3 cache plan ({split}) images"):
        image_path = resolve_sample_path(dataset_obj, dataset_index)
        image_size = resolve_image_size(dataset_obj, dataset_index)
        for spec in concept_specs:
            record = build_manifest_record(
                split=split,
                dataset_index=int(dataset_index),
                image_path=image_path,
                image_size=image_size,
                concept_index=int(spec.concept_index),
                concept=concepts[int(spec.concept_index)],
                backend=backend,
                prompt_template=prompt_template,
                status=status,
            )
            record["raw_concept"] = spec.raw_concept
            record["normalized_concept"] = spec.normalized_concept
            record["normalized_prompt"] = spec.normalized_prompt
            record["localizability"] = spec.localizability
            record["keep_for_masking"] = bool(spec.keep_for_masking)
            if backend == "groundingdino_sam3":
                record["prompt"] = spec.normalized_prompt
            records.append(record)
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
    concept_processing_cfg = dict(config.get("concept_processing", {}))
    concept_specs = build_grounding_specs(
        concepts=concepts,
        concept_indices=concept_indices,
        overrides=concept_processing_cfg.get("overrides"),
        allowed_localizability=concept_processing_cfg.get("allowed_localizability"),
    )
    if bool(concept_processing_cfg.get("filter_before_masking", False)):
        concept_specs = [spec for spec in concept_specs if spec.keep_for_masking]
        concept_indices = [spec.concept_index for spec in concept_specs]
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
        concept_specs=concept_specs,
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
                    prediction = runner.predict(str(image_path), str(record["prompt"]))
                    record["status"] = prediction["status"]
                    record["score"] = prediction.get("score")
                    record["num_detections"] = prediction.get("num_detections")
                    record["bbox_xyxy"] = prediction.get("bbox_xyxy")
                    record["area_ratio"] = prediction.get("area_ratio")
                    record["mask_selection"] = prediction.get("mask_selection")
                    record["selection_fallback_from"] = prediction.get("selection_fallback_from")
                    record["selected_source_index"] = prediction.get("selected_source_index")
                    record["selected_selection_metric"] = prediction.get("selected_selection_metric")
                    record["candidate_count"] = prediction.get("candidate_count", prediction.get("num_detections"))
                    record["grounding_score"] = prediction.get("grounding_score")
                    record["failure_reason"] = None if prediction["status"] == "ok" else prediction["status"]
                    if prediction["status"] == "ok":
                        mask_path = layout.root / split / str(record["mask_path"])
                        _write_mask_npz(mask_path, prediction)
                        _attach_candidate_preview_paths(
                            record=record,
                            prediction=prediction,
                            layout=layout,
                            split=split,
                            image_path=str(image_path),
                            preview_alpha=float(cache_cfg.get("preview_alpha", 0.45)),
                            write_candidate_previews=bool(cache_cfg.get("write_candidate_previews", True)),
                        )
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
            "concept_processing": concept_processing_cfg,
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
