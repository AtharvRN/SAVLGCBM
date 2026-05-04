from __future__ import annotations

import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_savlg_imagenet_standalone import (  # noqa: E402
    PREPROCESS_RESIZE_SIZE,
    PrecomputedTargetStore,
    resize_short_edge_size,
    transform_box_for_model_input,
)


def format_concept(text: str) -> str:
    text = text.lower()
    for ch in "-,.()":
        text = text.replace(ch, " ")
    if text.startswith("a "):
        text = text[2:]
    elif text.startswith("an "):
        text = text[3:]
    return " ".join(text.split())


def canonicalize_concept_label(text: str) -> str:
    return format_concept(text)


def slugify(text: str) -> str:
    pieces: List[str] = []
    for ch in text.lower():
        if ch.isalnum():
            pieces.append(ch)
        elif ch in {" ", "-", "_", "/"}:
            pieces.append("_")
    slug = "".join(pieces)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "item"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def safe_open_image(path: str, fallback_size: int = 224) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (FileNotFoundError, OSError, UnidentifiedImageError):
        return Image.new("RGB", (fallback_size, fallback_size), color=(0, 0, 0))


def get_image_size(path: str, fallback_size: int = 224) -> Tuple[int, int]:
    with safe_open_image(path, fallback_size=fallback_size) as image:
        return int(image.size[0]), int(image.size[1])


def read_concepts(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        concepts = [canonicalize_concept_label(line.strip()) for line in handle if line.strip()]
    return list(dict.fromkeys(concepts))


def _loadmat(path: Path) -> Dict[str, Any]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("scipy is required to read Stanford Cars .mat files") from exc
    return loadmat(str(path), squeeze_me=True, struct_as_record=False)


def _mat_get(obj: Any, key: str) -> Any:
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, np.void) and obj.dtype.names and key in obj.dtype.names:
        return obj[key]
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    raise KeyError(f"Could not find key={key!r} in MAT annotation object")


def _mat_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _mat_scalar(value.item())
        if value.size == 1:
            return _mat_scalar(value.reshape(-1)[0])
        return [_mat_scalar(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _mat_str(value: Any) -> str:
    value = _mat_scalar(value)
    if isinstance(value, list):
        if len(value) == 1:
            return _mat_str(value[0])
        return "".join(str(item) for item in value)
    return str(value)


def load_class_names(meta_path: Path) -> List[str]:
    payload = _loadmat(meta_path)
    raw = payload.get("class_names")
    if raw is None:
        raise KeyError(f"class_names missing from {meta_path}")
    names = [_mat_str(item).strip() for item in np.atleast_1d(raw).reshape(-1).tolist()]
    return [name for name in names if name]


def discover_stanford_cars_root(dataset_root: Path) -> Tuple[Path, str]:
    dataset_root = dataset_root.resolve()
    candidates = [dataset_root]
    candidates.extend(path for path in dataset_root.iterdir() if path.is_dir())
    for child in list(candidates):
        candidates.extend(path for path in child.iterdir() if path.is_dir())

    for candidate in candidates:
        if (candidate / "cars_train").is_dir() and (candidate / "cars_test").is_dir():
            return candidate, "official_split"
        if (candidate / "car_ims").is_dir() and (candidate / "cars_annos.mat").is_file():
            return candidate, "single_mat"
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate, "class_folders"
    raise FileNotFoundError(
        f"Could not detect a Stanford Cars layout under {dataset_root}. "
        "Expected official cars_train/cars_test, car_ims/cars_annos.mat, or train/test class folders."
    )


def _resolve_existing_path(root: Path, candidates: Sequence[str], *, kind: str) -> Path:
    for candidate in candidates:
        path = root / candidate
        if kind == "dir" and path.is_dir():
            return path
        if kind == "file" and path.is_file():
            return path
    raise FileNotFoundError(f"Could not resolve {kind} under {root} from candidates={list(candidates)}")


def _resolve_nested_image_dir(root: Path, base_name: str) -> Path:
    primary = root / base_name
    nested = primary / base_name
    if nested.is_dir():
        return nested
    if primary.is_dir():
        return primary
    raise FileNotFoundError(f"Could not find image directory for {base_name} under {root}")


def _official_test_annotation_path(root: Path) -> Path:
    candidates = [
        root / "cars_test_annos_withlabels.mat",
        root / "cars_test_annoswithlabels.mat",
        root / "devkit" / "cars_test_annos_withlabels.mat",
        root / "devkit" / "cars_test_annoswithlabels.mat",
        root / "devkit" / "cars_test_annos.mat",
        root / "car_devkit" / "devkit" / "cars_test_annos_withlabels.mat",
        root / "car_devkit" / "devkit" / "cars_test_annoswithlabels.mat",
        root / "car_devkit" / "devkit" / "cars_test_annos.mat",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find Stanford Cars test annotation MAT under {root}")


def _load_split_annotations(mat_path: Path, image_dir: Path, split: str, class_names: Sequence[str]) -> List[Dict[str, Any]]:
    payload = _loadmat(mat_path)
    annotations = payload.get("annotations")
    if annotations is None:
        raise KeyError(f"annotations missing from {mat_path}")
    rows: List[Dict[str, Any]] = []
    for idx, ann in enumerate(np.atleast_1d(annotations).reshape(-1).tolist()):
        fname = _mat_str(_mat_get(ann, "fname")).strip()
        class_raw = _mat_scalar(_mat_get(ann, "class")) if hasattr(ann, "class") or (isinstance(ann, np.void) and ann.dtype.names and "class" in ann.dtype.names) else None
        class_id = int(class_raw) - 1 if class_raw is not None else -1
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else "unknown"
        bbox = [
            float(_mat_scalar(_mat_get(ann, "bbox_x1"))),
            float(_mat_scalar(_mat_get(ann, "bbox_y1"))),
            float(_mat_scalar(_mat_get(ann, "bbox_x2"))),
            float(_mat_scalar(_mat_get(ann, "bbox_y2"))),
        ]
        image_path = (image_dir / fname).resolve()
        width, height = get_image_size(str(image_path))
        rows.append(
            {
                "path": str(image_path),
                "image_path": str(image_path),
                "relative_path": fname,
                "image_id": Path(fname).stem,
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "original_width": int(width),
                "original_height": int(height),
                "object_bbox_xyxy": bbox,
                "sample_index": idx,
            }
        )
    return rows


def _load_single_mat_annotations(root: Path) -> List[Dict[str, Any]]:
    meta_path = root / "cars_meta.mat"
    if not meta_path.is_file():
        raise FileNotFoundError(f"cars_meta.mat is required next to cars_annos.mat under {root}")
    class_names = load_class_names(meta_path)
    payload = _loadmat(root / "cars_annos.mat")
    annotations = payload.get("annotations")
    if annotations is None:
        raise KeyError(f"annotations missing from {root / 'cars_annos.mat'}")
    rows: List[Dict[str, Any]] = []
    counters = {"train": 0, "test": 0}
    for ann in np.atleast_1d(annotations).reshape(-1).tolist():
        fname = _mat_str(_mat_get(ann, "fname")).strip()
        class_id = int(_mat_scalar(_mat_get(ann, "class"))) - 1
        split = "test" if int(_mat_scalar(_mat_get(ann, "test"))) == 1 else "train"
        bbox = [
            float(_mat_scalar(_mat_get(ann, "bbox_x1"))),
            float(_mat_scalar(_mat_get(ann, "bbox_y1"))),
            float(_mat_scalar(_mat_get(ann, "bbox_x2"))),
            float(_mat_scalar(_mat_get(ann, "bbox_y2"))),
        ]
        image_path = (root / "car_ims" / fname).resolve()
        width, height = get_image_size(str(image_path))
        sample_index = counters[split]
        counters[split] += 1
        rows.append(
            {
                "path": str(image_path),
                "image_path": str(image_path),
                "relative_path": fname,
                "image_id": Path(fname).stem,
                "split": split,
                "class_id": class_id,
                "class_name": class_names[class_id],
                "original_width": int(width),
                "original_height": int(height),
                "object_bbox_xyxy": bbox,
                "sample_index": sample_index,
            }
        )
    return rows


def _load_class_folder_split(split_dir: Path, split: str, class_names: Sequence[str]) -> List[Dict[str, Any]]:
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    rows: List[Dict[str, Any]] = []
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        class_name = class_dir.name
        class_id = class_to_id[class_name]
        for image_idx, image_path in enumerate(sorted(class_dir.rglob("*"))):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            width, height = get_image_size(str(image_path))
            rel_path = image_path.relative_to(split_dir).as_posix()
            rows.append(
                {
                    "path": str(image_path.resolve()),
                    "image_path": str(image_path.resolve()),
                    "relative_path": rel_path,
                    "image_id": slugify(Path(rel_path).with_suffix("").as_posix()),
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "original_width": int(width),
                    "original_height": int(height),
                    "object_bbox_xyxy": None,
                    "sample_index": image_idx,
                }
            )
    return rows


def load_stanford_cars_records(dataset_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    root, layout = discover_stanford_cars_root(dataset_root)
    if layout == "official_split":
        devkit_dir = _resolve_existing_path(root, ["devkit", "car_devkit/devkit"], kind="dir")
        meta_path = _resolve_existing_path(devkit_dir, ["cars_meta.mat"], kind="file")
        train_ann_path = _resolve_existing_path(devkit_dir, ["cars_train_annos.mat"], kind="file")
        train_image_dir = _resolve_nested_image_dir(root, "cars_train")
        test_image_dir = _resolve_nested_image_dir(root, "cars_test")
        class_names = load_class_names(meta_path)
        return {
            "train": _load_split_annotations(train_ann_path, train_image_dir, "train", class_names),
            "test": _load_split_annotations(_official_test_annotation_path(root), test_image_dir, "test", class_names),
        }
    if layout == "single_mat":
        rows = _load_single_mat_annotations(root)
        out: Dict[str, List[Dict[str, Any]]] = {"train": [], "test": []}
        for row in rows:
            out[row["split"]].append(row)
        return out
    if layout == "class_folders":
        train_dir = root / "train"
        test_dir = root / "test"
        class_names = sorted(
            {path.name for path in train_dir.iterdir() if path.is_dir()} | {path.name for path in test_dir.iterdir() if path.is_dir()}
        )
        return {
            "train": _load_class_folder_split(train_dir, "train", class_names),
            "test": _load_class_folder_split(test_dir, "test", class_names),
        }
    raise ValueError(f"Unsupported Stanford Cars layout: {layout}")


def stratified_split_train_val(
    train_rows: Sequence[Dict[str, Any]],
    *,
    val_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in train_rows:
        grouped.setdefault(int(row["class_id"]), []).append(dict(row))

    rng = random.Random(seed)
    train_split: List[Dict[str, Any]] = []
    val_split: List[Dict[str, Any]] = []
    for class_id in sorted(grouped):
        items = grouped[class_id]
        rng.shuffle(items)
        n_items = len(items)
        n_val = int(round(n_items * float(val_fraction)))
        n_val = max(1, n_val)
        n_val = min(n_val, max(n_items - 1, 1))
        val_items = items[:n_val]
        train_items = items[n_val:]
        if not train_items:
            train_items = val_items[-1:]
            val_items = val_items[:-1]
        for row in train_items:
            row["split"] = "train"
        for row in val_items:
            row["split"] = "val"
        train_split.extend(train_items)
        val_split.extend(val_items)

    train_split.sort(key=lambda row: (int(row["class_id"]), str(row["relative_path"])))
    val_split.sort(key=lambda row: (int(row["class_id"]), str(row["relative_path"])))
    for split_rows in (train_split, val_split):
        for sample_index, row in enumerate(split_rows):
            row["sample_index"] = sample_index
    return train_split, val_split


def summarize_manifest(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    classes = sorted({int(row["class_id"]) for row in rows if int(row["class_id"]) >= 0})
    bbox_count = sum(1 for row in rows if row.get("object_bbox_xyxy"))
    return {
        "n_images": len(rows),
        "n_classes": len(classes),
        "bbox_count": bbox_count,
    }


def annotation_file_path(annotation_root: Path, split: str, image_id: str) -> Path:
    return annotation_root / split / f"{image_id}.json"


def load_annotation_payload(annotation_root: Path, split: str, image_id: str) -> Dict[str, Any]:
    path = annotation_file_path(annotation_root, split, image_id)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {"concepts": payload}


def annotation_concepts_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    concepts = payload.get("concepts", [])
    if isinstance(concepts, list):
        return [item for item in concepts if isinstance(item, dict)]
    return []


class StanfordCarsManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        annotation_dir: str,
        concepts: Sequence[str],
        split: str,
        input_size: int,
        min_image_bytes: int,
        train_random_transforms: bool = False,
    ) -> None:
        self.manifest_path = str(manifest_path)
        self.annotation_dir = str(annotation_dir)
        self.split = str(split)
        self.input_size = int(input_size)
        self.min_image_bytes = int(min_image_bytes)
        self.train_random_transforms = bool(train_random_transforms)
        self.concepts = list(concepts)
        self.concept_to_idx = {name: idx for idx, name in enumerate(self.concepts)}
        self.records = load_jsonl(Path(manifest_path))
        if not self.records:
            raise ValueError(f"Manifest is empty: {manifest_path}")
        self.precomputed_targets: Optional[PrecomputedTargetStore] = None
        self.sample_indices = None
        samples = [(str(row["image_path"]), int(row["class_id"])) for row in self.records]
        class_names: Dict[int, str] = {}
        for row in self.records:
            class_names[int(row["class_id"])] = str(row["class_name"])
        max_class_id = max(class_names)
        classes = [str(idx) for idx in range(max_class_id + 1)]
        for class_id, class_name in class_names.items():
            classes[class_id] = class_name
        self.dataset = SimpleNamespace(
            samples=samples,
            classes=classes,
            transform=self._transform(split),
        )

    def _transform(self, split: str) -> transforms.Compose:
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        if split == "train" and self.train_random_transforms:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(PREPROCESS_RESIZE_SIZE),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def _safe_loader(self, path: str) -> Image.Image:
        try:
            if os.path.getsize(path) < self.min_image_bytes:
                raise OSError(f"tiny file: {path}")
            with Image.open(path) as image:
                return image.convert("RGB")
        except (FileNotFoundError, OSError, UnidentifiedImageError):
            return Image.new("RGB", (self.input_size, self.input_size), color=0)

    def attach_precomputed_targets(self, root: str, cfg: Optional[Any] = None) -> None:
        target_dir = Path(root) / self.split
        if not target_dir.is_dir():
            raise FileNotFoundError(f"Missing precomputed target directory: {target_dir}")
        self.precomputed_targets = PrecomputedTargetStore(target_dir)
        if len(self.precomputed_targets) != len(self.records):
            raise ValueError(
                f"Precomputed targets at {target_dir} have {len(self.precomputed_targets)} entries, "
                f"expected {len(self.records)}"
            )
        if self.precomputed_targets.n_concepts != len(self.concepts):
            raise ValueError(
                f"Precomputed targets at {target_dir} have {self.precomputed_targets.n_concepts} concepts, "
                f"expected {len(self.concepts)}"
            )
        if cfg is not None:
            self.precomputed_targets.validate_target_geometry(cfg)

    def apply_concept_filter(self, keep_indices: Sequence[int]) -> None:
        keep = [int(idx) for idx in keep_indices]
        self.concepts = [self.concepts[idx] for idx in keep]
        self.concept_to_idx = {name: idx for idx, name in enumerate(self.concepts)}
        if self.precomputed_targets is not None:
            self.precomputed_targets.set_concept_filter(keep)

    def _load_annotation(self, index: int) -> List[Dict[str, Any]]:
        row = self.records[int(index)]
        payload = load_annotation_payload(Path(self.annotation_dir), self.split, str(row["image_id"]))
        return annotation_concepts_from_payload(payload)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.records[int(index)]
        image_path = str(row["image_path"])
        class_id = int(row["class_id"])
        with self._safe_loader(image_path) as raw_image:
            image_size = (int(raw_image.size[0]), int(raw_image.size[1]))
            image = self.dataset.transform(raw_image) if self.dataset.transform is not None else raw_image
        item: Dict[str, Any] = {
            "image": image,
            "class_id": class_id,
            "sample_index": int(index),
            "image_size": image_size,
            "image_id": str(row["image_id"]),
            "image_path": image_path,
        }
        if self.precomputed_targets is not None:
            item.update(self.precomputed_targets.get(int(index)))
        else:
            item["annotation"] = self._load_annotation(int(index))
        return item


def _xyxy_from_annotation(annotation: Dict[str, Any]) -> Optional[List[float]]:
    box = annotation.get("box_xyxy")
    if box is None:
        box = annotation.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    return [float(value) for value in box]


def render_box_audit(
    image_path: str,
    annotations: Sequence[Dict[str, Any]],
    output_path: Path,
    *,
    input_size: int = 224,
    resize_size: int = PREPROCESS_RESIZE_SIZE,
    max_boxes: int = 12,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with safe_open_image(image_path, fallback_size=input_size) as image:
        original = image.copy()
    width, height = original.size
    original_draw = ImageDraw.Draw(original)

    resized_width, resized_height = resize_short_edge_size((width, height), resize_size=resize_size)
    resized = original.resize((resized_width, resized_height), resample=Image.BILINEAR)
    crop_left = max(int(round((resized_width - input_size) / 2.0)), 0)
    crop_top = max(int(round((resized_height - input_size) / 2.0)), 0)
    crop = resized.crop((crop_left, crop_top, crop_left + input_size, crop_top + input_size))
    crop_draw = ImageDraw.Draw(crop)

    rendered = 0
    kept_labels: List[str] = []
    for annotation in annotations:
        if rendered >= max_boxes:
            break
        box_xyxy = _xyxy_from_annotation(annotation)
        if box_xyxy is None:
            continue
        label = str(annotation.get("canonical_label") or annotation.get("label") or f"box_{rendered}")
        original_draw.rectangle(box_xyxy, outline="red", width=3)
        original_draw.text((box_xyxy[0] + 2.0, box_xyxy[1] + 2.0), label, fill="red")
        transformed = transform_box_for_model_input(
            box_xyxy,
            image_size=(width, height),
            input_size=input_size,
            resize_size=resize_size,
        )
        if transformed is not None:
            tx1, ty1, tx2, ty2 = transformed
            crop_box = [tx1 * input_size, ty1 * input_size, tx2 * input_size, ty2 * input_size]
            crop_draw.rectangle(crop_box, outline="lime", width=3)
            crop_draw.text((crop_box[0] + 2.0, crop_box[1] + 2.0), label, fill="lime")
        kept_labels.append(label)
        rendered += 1

    canvas = Image.new("RGB", (original.width + crop.width, max(original.height, crop.height)), color=(255, 255, 255))
    canvas.paste(original, (0, 0))
    canvas.paste(crop, (original.width, 0))
    canvas.save(output_path)
    return {
        "image_path": str(image_path),
        "output_path": str(output_path),
        "rendered_boxes": rendered,
        "labels": kept_labels,
    }


def iter_manifest_rows(manifest_paths: Iterable[Path]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for manifest_path in manifest_paths:
        split = manifest_path.stem.replace("_manifest", "")
        for row in load_jsonl(manifest_path):
            yield split, row
