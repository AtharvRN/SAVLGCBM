"""Utilities and schema helpers for SAM3 concept pseudo-mask caches.

The cache is intentionally independent from CUB part mappings. It is keyed by
dataset image index and concept index from the concept set supplied to the
generator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from PIL import Image


CACHE_SCHEMA_VERSION = "sam3_concept_masks_v1"
MASK_FORMAT = "npz_bool_hxw_v1"


def canonicalize_concept_label(s: str) -> str:
    normalized = s.lower()
    for ch in "-,.()":
        normalized = normalized.replace(ch, " ")
    normalized = " ".join(normalized.split())
    if normalized.startswith("a "):
        normalized = normalized[2:]
    elif normalized.startswith("an "):
        normalized = normalized[3:]
    return " ".join(normalized.split())


@dataclass(frozen=True)
class Sam3CacheLayout:
    """Resolved paths for one dataset split of one SAM3 concept-mask cache."""

    root: Path
    split: str

    @property
    def manifest_path(self) -> Path:
        return self.root / self.split / "manifest.json"

    @property
    def masks_dir(self) -> Path:
        return self.root / self.split / "masks"

    @property
    def metadata_dir(self) -> Path:
        return self.root / self.split / "metadata"

    @property
    def previews_dir(self) -> Path:
        return self.root / self.split / "previews"

    @property
    def candidate_previews_dir(self) -> Path:
        return self.root / self.split / "candidate_previews"

    def ensure_dirs(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_previews_dir.mkdir(parents=True, exist_ok=True)


def concept_hash(concepts: Sequence[str]) -> str:
    return hashlib.sha1("\n".join(concepts).encode("utf-8")).hexdigest()[:16]


def load_concept_set(path: str, max_concepts: Optional[int] = None) -> List[str]:
    with open(path, "r") as f:
        concepts = [
            canonicalize_concept_label(line.strip())
            for line in f
            if line.strip()
        ]
    concepts = list(dict.fromkeys(concepts))
    if max_concepts is not None:
        concepts = concepts[: int(max_concepts)]
    return concepts


def select_indices(
    total: int,
    explicit_indices: Optional[Sequence[int]] = None,
    max_items: Optional[int] = None,
) -> List[int]:
    if explicit_indices:
        selected = [int(idx) for idx in explicit_indices]
    else:
        selected = list(range(int(total)))
    if max_items is not None:
        selected = selected[: int(max_items)]
    for idx in selected:
        if idx < 0 or idx >= total:
            raise IndexError(f"Index {idx} is outside dataset length {total}")
    return selected


def resolve_sample_path(dataset: Any, dataset_index: int) -> Optional[str]:
    samples = getattr(dataset, "samples", None) or getattr(dataset, "imgs", None)
    if samples is None:
        return None
    return str(samples[int(dataset_index)][0])


def resolve_image_size(dataset: Any, dataset_index: int) -> List[int]:
    sample_path = resolve_sample_path(dataset, dataset_index)
    if sample_path is not None:
        with Image.open(sample_path) as img:
            return [int(img.size[0]), int(img.size[1])]
    image, _ = dataset[int(dataset_index)]
    return [int(image.size[0]), int(image.size[1])]


def _record_stem(dataset_index: int, concept_index: int) -> str:
    return f"image_{int(dataset_index):06d}/concept_{int(concept_index):04d}"


def relative_record_paths(dataset_index: int, concept_index: int) -> Dict[str, str]:
    stem = _record_stem(dataset_index, concept_index)
    return {
        "mask_path": f"masks/{stem}.npz",
        "metadata_path": f"metadata/{stem}.json",
        "preview_path": f"previews/{stem}.png",
    }


def relative_candidate_preview_path(dataset_index: int, concept_index: int, candidate_rank: int) -> str:
    stem = _record_stem(dataset_index, concept_index)
    return f"candidate_previews/{stem}/candidate_{int(candidate_rank):02d}.png"


def build_manifest_record(
    split: str,
    dataset_index: int,
    image_path: Optional[str],
    image_size: Sequence[int],
    concept_index: int,
    concept: str,
    backend: str,
    prompt_template: str,
    status: str = "pending",
) -> Dict[str, Any]:
    paths = relative_record_paths(dataset_index, concept_index)
    prompt = prompt_template.format(concept=concept)
    return {
        "record_id": f"{split}/{int(dataset_index):06d}/{int(concept_index):04d}",
        "dataset_index": int(dataset_index),
        "image_path": image_path,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "concept_index": int(concept_index),
        "concept": concept,
        "prompt": prompt,
        "status": status,
        "score": None,
        "mask_format": MASK_FORMAT,
        "mask_path": paths["mask_path"],
        "metadata_path": paths["metadata_path"],
        "preview_path": paths["preview_path"],
        "backend": backend,
        "failure_reason": None,
    }


def build_manifest(
    dataset: str,
    split: str,
    concept_set: str,
    concepts: Sequence[str],
    records: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    selected_concept_indices = sorted(
        {int(record["concept_index"]) for record in records}
    )
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": dataset,
        "split": split,
        "concept_set": concept_set,
        "concept_hash": concept_hash(concepts),
        "selected_concept_indices": selected_concept_indices,
        "mask_format": MASK_FORMAT,
        "image_count": len({int(record["dataset_index"]) for record in records}),
        "concept_count": len(concepts),
        "record_count": len(records),
        "records": list(records),
        "config": config,
        "notes": [
            "Concepts come directly from concept_set; no CUB concept-to-part filter is applied.",
            "Masks are expected in original image pixel coordinates as bool/uint8 HxW arrays.",
        ],
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_manifest(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        payload = json.load(f)
    version = payload.get("schema_version")
    if version != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported SAM3 concept-mask cache schema {version!r}; expected {CACHE_SCHEMA_VERSION!r}"
        )
    return payload


def iter_records(manifest: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    yield from manifest.get("records", [])
