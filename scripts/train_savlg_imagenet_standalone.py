import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights, resnet50


IMAGENET_LABEL_ALIASES = {
    "website": "a web page",
    "beer bottle": "a bottle with a long neck",
    "wine bottle": "a bottle with a long neck",
    "soda bottle": "a glass or plastic bottle",
    "ski": "a pair of skis",
    "metal nail": "nails",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone ImageNet SAVLG trainer optimized for A10 throughput."
    )
    parser.add_argument("--mode", choices=["train", "profile"], default="train")
    parser.add_argument("--train_root", required=True)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--concept_file", default="concept_files/imagenet_filtered.txt")
    parser.add_argument("--val_root", default="")
    parser.add_argument("--save_dir", default="saved_models/imagenet_standalone")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_val_images", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=["fp16", "bf16", "none"], default="fp16")
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--disable_channels_last", action="store_true")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--cudnn_benchmark", action="store_true", default=True)
    parser.add_argument("--disable_cudnn_benchmark", action="store_true")
    parser.add_argument("--seed", type=int, default=6885)
    parser.add_argument("--min_image_bytes", type=int, default=2048)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--mask_h", type=int, default=7)
    parser.add_argument("--mask_w", type=int, default=7)
    parser.add_argument("--patch_iou_thresh", type=float, default=0.5)
    parser.add_argument("--concept_threshold", type=float, default=0.15)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--global_pos_weight", type=float, default=1.0)
    parser.add_argument("--patch_pos_weight", type=float, default=1.0)
    parser.add_argument("--loss_global_w", type=float, default=1.0)
    parser.add_argument("--loss_mask_w", type=float, default=1.0)
    parser.add_argument("--loss_dice_w", type=float, default=0.0)
    parser.add_argument("--branch_arch", choices=["shared", "dual"], default="shared")
    parser.add_argument(
        "--spatial_branch_mode", choices=["shared_stage", "multiscale_conv45"], default="shared_stage"
    )
    parser.add_argument("--spatial_stage", choices=["conv4", "conv5"], default="conv5")
    parser.add_argument("--residual_alpha", type=float, default=0.8)
    parser.add_argument("--profile_steps", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--print_config", action="store_true")
    return parser.parse_args()


@dataclass
class Config:
    mode: str
    train_root: str
    annotation_dir: str
    concept_file: str
    val_root: str
    save_dir: str
    run_name: str
    max_train_images: int
    max_val_images: int
    val_split: float
    epochs: int
    batch_size: int
    workers: int
    prefetch_factor: int
    persistent_workers: bool
    pin_memory: bool
    device: str
    amp: str
    channels_last: bool
    tf32: bool
    cudnn_benchmark: bool
    seed: int
    min_image_bytes: int
    input_size: int
    mask_h: int
    mask_w: int
    patch_iou_thresh: float
    concept_threshold: float
    optimizer: str
    lr: float
    weight_decay: float
    momentum: float
    global_pos_weight: float
    patch_pos_weight: float
    loss_global_w: float
    loss_mask_w: float
    loss_dice_w: float
    branch_arch: str
    spatial_branch_mode: str
    spatial_stage: str
    residual_alpha: float
    profile_steps: int
    warmup_steps: int
    log_every: int
    save_every: int


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        mode=args.mode,
        train_root=args.train_root,
        annotation_dir=args.annotation_dir,
        concept_file=args.concept_file,
        val_root=args.val_root,
        save_dir=args.save_dir,
        run_name=args.run_name,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        val_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        prefetch_factor=max(1, int(args.prefetch_factor)),
        persistent_workers=bool(args.persistent_workers and not args.disable_persistent_workers),
        pin_memory=bool(args.pin_memory and not args.disable_pin_memory),
        device=args.device,
        amp=args.amp,
        channels_last=bool(args.channels_last and not args.disable_channels_last),
        tf32=bool(args.tf32 and not args.disable_tf32),
        cudnn_benchmark=bool(args.cudnn_benchmark and not args.disable_cudnn_benchmark),
        seed=args.seed,
        min_image_bytes=args.min_image_bytes,
        input_size=args.input_size,
        mask_h=args.mask_h,
        mask_w=args.mask_w,
        patch_iou_thresh=args.patch_iou_thresh,
        concept_threshold=args.concept_threshold,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        global_pos_weight=args.global_pos_weight,
        patch_pos_weight=args.patch_pos_weight,
        loss_global_w=args.loss_global_w,
        loss_mask_w=args.loss_mask_w,
        loss_dice_w=args.loss_dice_w,
        branch_arch=args.branch_arch,
        spatial_branch_mode=args.spatial_branch_mode,
        spatial_stage=args.spatial_stage,
        residual_alpha=args.residual_alpha,
        profile_steps=args.profile_steps,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        save_every=args.save_every,
    )


def configure_runtime(cfg: Config) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def amp_dtype(name: str) -> Optional[torch.dtype]:
    if name == "none":
        return None
    if name == "bf16":
        return torch.bfloat16
    return torch.float16


def autocast_context(cfg: Config):
    dtype = amp_dtype(cfg.amp)
    if dtype is None or not str(cfg.device).startswith("cuda"):
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=dtype)


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
    normalized = format_concept(text)
    return format_concept(IMAGENET_LABEL_ALIASES.get(normalized, normalized))


def load_concepts(path: str) -> List[str]:
    with open(path, "r") as handle:
        concepts = [canonicalize_concept_label(line.strip()) for line in handle if line.strip()]
    return list(dict.fromkeys(concepts))


class SafeImageFolderWithAnnotations(Dataset):
    def __init__(
        self,
        root: str,
        annotation_dir: str,
        concepts: Sequence[str],
        input_size: int,
        min_image_bytes: int,
        split: str,
    ) -> None:
        self.root = root
        self.annotation_dir = annotation_dir
        self.input_size = int(input_size)
        self.min_image_bytes = int(min_image_bytes)
        self.split = split
        self.concepts = list(concepts)
        self.concept_to_idx = {name: idx for idx, name in enumerate(self.concepts)}
        self.dataset = ImageFolder(root=root, loader=self._safe_loader, transform=self._transform(split))

    def _transform(self, split: str) -> transforms.Compose:
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        if split == "train":
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
                transforms.Resize(256),
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

    def _annotation_path(self, sample_index: int) -> Path:
        split_dir = "imagenet_train" if self.split == "train" else "imagenet_val"
        return Path(self.annotation_dir) / split_dir / f"{sample_index}.json"

    def _load_annotation(self, sample_index: int) -> List[Dict[str, Any]]:
        path = self._annotation_path(sample_index)
        if not path.exists():
            return []
        try:
            with path.open("r") as handle:
                payload = json.load(handle)
        except Exception:
            return []
        if isinstance(payload, list):
            return payload
        return payload.get("concepts", [])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        path, class_id = self.dataset.samples[index]
        with self._safe_loader(path) as raw_image:
            image_size = (int(raw_image.size[0]), int(raw_image.size[1]))
            image = self.dataset.transform(raw_image) if self.dataset.transform is not None else raw_image
        return {
            "image": image,
            "class_id": int(class_id),
            "sample_index": int(index),
            "annotation": self._load_annotation(index),
            "image_size": image_size,
        }


class DatasetView(Dataset):
    def __init__(self, base_dataset: SafeImageFolderWithAnnotations, indices: Sequence[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.concepts = base_dataset.concepts
        self.concept_to_idx = base_dataset.concept_to_idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.base_dataset[self.indices[index]]


def split_train_val(
    dataset: SafeImageFolderWithAnnotations,
    val_split: float,
    max_train_images: int,
    max_val_images: int,
    seed: int,
) -> Tuple[DatasetView, DatasetView]:
    total = len(dataset)
    indices = list(range(total))
    if max_train_images > 0:
        total = min(total, max_train_images)
        indices = indices[:total]
    n_val = int(round(float(val_split) * len(indices)))
    n_val = min(max(n_val, 1), max(len(indices) - 1, 1))
    generator = random.Random(seed)
    generator.shuffle(indices)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    if max_val_images > 0:
        val_indices = val_indices[:max_val_images]
    return DatasetView(dataset, train_indices), DatasetView(dataset, val_indices)


def collate_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "class_ids": torch.tensor([item["class_id"] for item in batch], dtype=torch.long),
        "sample_indices": torch.tensor([item["sample_index"] for item in batch], dtype=torch.long),
        "annotations": [item["annotation"] for item in batch],
        "image_sizes": [item["image_size"] for item in batch],
    }


def build_loader(dataset: Dataset, cfg: Config, shuffle: bool, drop_last: bool) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": cfg.batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.workers,
        "pin_memory": cfg.pin_memory,
        "collate_fn": collate_batch,
        "drop_last": drop_last,
    }
    if cfg.workers > 0:
        kwargs["persistent_workers"] = cfg.persistent_workers
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    return DataLoader(**kwargs)


def normalize_box(box: Sequence[float], image_size: Tuple[int, int]) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    width, height = image_size
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        if width <= 0 or height <= 0:
            return None
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
    x1, x2 = sorted((float(np.clip(x1, 0.0, 1.0)), float(np.clip(x2, 0.0, 1.0))))
    y1, y2 = sorted((float(np.clip(y1, 0.0, 1.0)), float(np.clip(y2, 0.0, 1.0))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def rasterize_box_iou(
    box: Sequence[float],
    image_size: Tuple[int, int],
    mask_h: int,
    mask_w: int,
    iou_thresh: float,
) -> Optional[np.ndarray]:
    norm = normalize_box(box, image_size=image_size)
    if norm is None:
        return None
    x1, y1, x2, y2 = norm
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if box_area <= 0.0:
        return None
    mask = np.zeros((mask_h, mask_w), dtype=np.float32)
    patch_area = 1.0 / float(mask_h * mask_w)
    for r in range(mask_h):
        py1 = r / float(mask_h)
        py2 = (r + 1) / float(mask_h)
        for c in range(mask_w):
            px1 = c / float(mask_w)
            px2 = (c + 1) / float(mask_w)
            ix1 = max(px1, x1)
            iy1 = max(py1, y1)
            ix2 = min(px2, x2)
            iy2 = min(py2, y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0.0:
                continue
            union = patch_area + box_area - inter
            if union > 0.0 and inter / union > iou_thresh:
                mask[r, c] = 1.0
    return mask


def build_gdino_targets(
    annotations: Sequence[List[Dict[str, Any]]],
    image_sizes: Sequence[Tuple[int, int]],
    concept_to_idx: Dict[str, int],
    n_concepts: int,
    cfg: Config,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    global_targets = torch.zeros((len(annotations), n_concepts), dtype=torch.float32)
    sparse_indices: List[torch.Tensor] = []
    sparse_masks: List[torch.Tensor] = []
    for sample_idx, sample_annotations in enumerate(annotations):
        scores = np.zeros((n_concepts,), dtype=np.float32)
        mask_dict: Dict[int, np.ndarray] = {}
        entries = sample_annotations[1:] if isinstance(sample_annotations, list) else []
        for ann in entries:
            if not isinstance(ann, dict):
                continue
            label = ann.get("label")
            if not isinstance(label, str):
                continue
            concept_idx = concept_to_idx.get(canonicalize_concept_label(label))
            if concept_idx is None:
                continue
            score = float(ann.get("logit", 0.0))
            if score > scores[concept_idx]:
                scores[concept_idx] = score
            if score < cfg.concept_threshold:
                continue
            mask = rasterize_box_iou(
                ann.get("box"),
                image_size=image_sizes[sample_idx],
                mask_h=cfg.mask_h,
                mask_w=cfg.mask_w,
                iou_thresh=cfg.patch_iou_thresh,
            )
            if mask is None:
                continue
            existing = mask_dict.get(concept_idx)
            if existing is None:
                mask_dict[concept_idx] = mask
            else:
                np.maximum(existing, mask, out=existing)
        global_targets[sample_idx] = torch.from_numpy((scores > cfg.concept_threshold).astype(np.float32))
        if mask_dict:
            keys = sorted(mask_dict.keys())
            sparse_indices.append(torch.tensor(keys, dtype=torch.long))
            sparse_masks.append(torch.from_numpy(np.stack([mask_dict[k] for k in keys], axis=0)))
        else:
            sparse_indices.append(torch.zeros((0,), dtype=torch.long))
            sparse_masks.append(torch.zeros((0, cfg.mask_h, cfg.mask_w), dtype=torch.float32))

    max_k = max((tensor.numel() for tensor in sparse_indices), default=0)
    if max_k == 0:
        idx_pad = torch.full((len(annotations), 1), -1, dtype=torch.long)
        mask_pad = torch.zeros((len(annotations), 1, cfg.mask_h, cfg.mask_w), dtype=torch.float32)
        valid = torch.zeros((len(annotations), 1), dtype=torch.bool)
    else:
        idx_pad = torch.full((len(annotations), max_k), -1, dtype=torch.long)
        mask_pad = torch.zeros((len(annotations), max_k, cfg.mask_h, cfg.mask_w), dtype=torch.float32)
        valid = torch.zeros((len(annotations), max_k), dtype=torch.bool)
        for batch_idx, (indices, masks) in enumerate(zip(sparse_indices, sparse_masks)):
            if indices.numel() == 0:
                continue
            idx_pad[batch_idx, : indices.numel()] = indices
            mask_pad[batch_idx, : indices.numel()] = masks
            valid[batch_idx, : indices.numel()] = True

    return (
        global_targets.to(device, non_blocking=True),
        idx_pad.to(device, non_blocking=True),
        mask_pad.to(device, non_blocking=True),
        valid.to(device, non_blocking=True),
    )


class ResNet50Conv45(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        conv4 = self.layer3(x)
        conv5 = self.layer4(conv4)
        return {"conv4": conv4, "conv5": conv5}


class SharedConceptHead(nn.Module):
    def __init__(self, n_concepts: int, spatial_stage: str) -> None:
        super().__init__()
        in_channels = 1024 if spatial_stage == "conv4" else 2048
        self.spatial_stage = spatial_stage
        self.spatial = nn.Conv2d(in_channels, n_concepts, kernel_size=1, bias=True)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        spatial_maps = self.spatial(feats[self.spatial_stage])
        pooled = F.adaptive_avg_pool2d(spatial_maps, 1).flatten(1)
        return {
            "global_logits": pooled,
            "spatial_logits": torch.zeros_like(pooled),
            "spatial_maps": spatial_maps,
            "final_logits": pooled,
        }


class DualBranchConceptHead(nn.Module):
    def __init__(self, n_concepts: int, spatial_stage: str, residual_alpha: float) -> None:
        super().__init__()
        in_channels = 1024 if spatial_stage == "conv4" else 2048
        self.spatial_stage = spatial_stage
        self.residual_alpha = residual_alpha
        self.global_head = nn.Linear(2048, n_concepts, bias=True)
        self.spatial = nn.Conv2d(in_channels, n_concepts, kernel_size=1, bias=True)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        global_logits = self.global_head(feats["conv5"].mean(dim=(2, 3)))
        spatial_maps = self.spatial(feats[self.spatial_stage])
        spatial_logits = spatial_maps.flatten(2).mean(dim=-1)
        final_logits = global_logits + self.residual_alpha * spatial_logits
        return {
            "global_logits": global_logits,
            "spatial_logits": spatial_logits,
            "spatial_maps": spatial_maps,
            "final_logits": final_logits,
        }


class MultiScaleDualBranchConceptHead(nn.Module):
    def __init__(self, n_concepts: int, residual_alpha: float, fusion_dim: int = 2048) -> None:
        super().__init__()
        self.residual_alpha = residual_alpha
        self.global_head = nn.Linear(2048, n_concepts, bias=True)
        self.conv4_proj = nn.Conv2d(1024, fusion_dim, kernel_size=1, bias=False)
        self.conv5_proj = nn.Conv2d(2048, fusion_dim, kernel_size=1, bias=False)
        self.spatial = nn.Conv2d(fusion_dim, n_concepts, kernel_size=1, bias=True)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        global_logits = self.global_head(feats["conv5"].mean(dim=(2, 3)))
        conv5_up = F.interpolate(
            self.conv5_proj(feats["conv5"]),
            size=feats["conv4"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = F.relu(self.conv4_proj(feats["conv4"]) + conv5_up, inplace=False)
        spatial_maps = self.spatial(fused)
        spatial_logits = spatial_maps.flatten(2).mean(dim=-1)
        final_logits = global_logits + self.residual_alpha * spatial_logits
        return {
            "global_logits": global_logits,
            "spatial_logits": spatial_logits,
            "spatial_maps": spatial_maps,
            "final_logits": final_logits,
        }


def build_model(cfg: Config, n_concepts: int) -> Tuple[nn.Module, nn.Module]:
    backbone = ResNet50Conv45().to(cfg.device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    backbone.eval()
    if cfg.channels_last:
        backbone.to(memory_format=torch.channels_last)
    if cfg.spatial_branch_mode == "multiscale_conv45":
        if cfg.branch_arch != "dual":
            raise ValueError("multiscale_conv45 requires branch_arch=dual")
        head = MultiScaleDualBranchConceptHead(n_concepts=n_concepts, residual_alpha=cfg.residual_alpha)
    elif cfg.branch_arch == "dual":
        head = DualBranchConceptHead(
            n_concepts=n_concepts,
            spatial_stage=cfg.spatial_stage,
            residual_alpha=cfg.residual_alpha,
        )
    else:
        head = SharedConceptHead(n_concepts=n_concepts, spatial_stage=cfg.spatial_stage)
    head = head.to(cfg.device)
    if cfg.channels_last:
        head.to(memory_format=torch.channels_last)
    return backbone, head


def prepare_images(images: torch.Tensor, cfg: Config) -> torch.Tensor:
    if cfg.channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images.to(cfg.device, non_blocking=cfg.pin_memory)


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    global_targets: torch.Tensor,
    mask_indices: torch.Tensor,
    mask_targets: torch.Tensor,
    mask_valid: torch.Tensor,
    cfg: Config,
) -> Dict[str, torch.Tensor]:
    final_logits = outputs["final_logits"]
    spatial_maps = F.interpolate(
        outputs["spatial_maps"],
        size=mask_targets.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    global_loss_raw = F.binary_cross_entropy_with_logits(final_logits, global_targets, reduction="none")
    global_pos_w = torch.where(
        global_targets > 0.5,
        torch.full_like(global_targets, float(cfg.global_pos_weight)),
        torch.ones_like(global_targets),
    )
    loss_global = (global_loss_raw * global_pos_w).sum() / torch.clamp(global_pos_w.sum(), min=1.0)

    per_sample_mask_losses: List[torch.Tensor] = []
    per_sample_dice_losses: List[torch.Tensor] = []
    for batch_index in range(spatial_maps.shape[0]):
        valid = mask_valid[batch_index]
        if not bool(valid.any()):
            continue
        concept_ids = mask_indices[batch_index][valid]
        pred = spatial_maps[batch_index].index_select(0, concept_ids)
        tgt = mask_targets[batch_index][valid].to(pred.dtype)

        bce_raw = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
        patch_pos_w = torch.where(
            tgt > 0.5,
            torch.full_like(tgt, float(cfg.patch_pos_weight)),
            torch.ones_like(tgt),
        )
        patch_pos_w_flat = patch_pos_w.flatten(1)
        per_concept_mask = (bce_raw.flatten(1) * patch_pos_w_flat).sum(dim=1) / torch.clamp(
            patch_pos_w_flat.sum(dim=1),
            min=1.0,
        )
        per_sample_mask_losses.append(per_concept_mask.mean())

        if cfg.loss_dice_w > 0.0:
            pred_prob = torch.sigmoid(pred).flatten(1)
            tgt_flat = tgt.flatten(1)
            intersection = (pred_prob * tgt_flat).sum(dim=1)
            denom = pred_prob.sum(dim=1) + tgt_flat.sum(dim=1)
            per_concept_dice = 1.0 - ((2.0 * intersection + 1e-6) / (denom + 1e-6))
            per_sample_dice_losses.append(per_concept_dice.mean())

    loss_mask = (
        torch.stack(per_sample_mask_losses).mean() if per_sample_mask_losses else spatial_maps.sum() * 0.0
    )
    loss_dice = (
        torch.stack(per_sample_dice_losses).mean() if per_sample_dice_losses else spatial_maps.sum() * 0.0
    )
    total = cfg.loss_global_w * loss_global + cfg.loss_mask_w * loss_mask + cfg.loss_dice_w * loss_dice
    return {
        "total": total,
        "global": loss_global.detach(),
        "mask": loss_mask.detach(),
        "dice": loss_dice.detach(),
    }


def make_optimizer(head: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(
        parameters,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum,
    )


def train_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: Config,
    concept_to_idx: Dict[str, int],
    n_concepts: int,
    epoch: int,
) -> Dict[str, float]:
    head.train()
    totals = {"total": 0.0, "global": 0.0, "mask": 0.0, "dice": 0.0, "count": 0}
    start_time = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        images = prepare_images(batch["images"], cfg)
        global_targets, idx_pad, mask_pad, valid_pad = build_gdino_targets(
            batch["annotations"],
            batch["image_sizes"],
            concept_to_idx,
            n_concepts,
            cfg,
            cfg.device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            feats = backbone(images)
        with autocast_context(cfg):
            outputs = head(feats)
            losses = compute_losses(outputs, global_targets, idx_pad, mask_pad, valid_pad, cfg)
            loss = losses["total"]
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = int(images.shape[0])
        totals["total"] += float(loss.detach().item()) * batch_size
        totals["global"] += float(losses["global"].item()) * batch_size
        totals["mask"] += float(losses["mask"].item()) * batch_size
        totals["dice"] += float(losses["dice"].item()) * batch_size
        totals["count"] += batch_size
        if step % cfg.log_every == 0:
            elapsed = time.perf_counter() - start_time
            images_per_second = totals["count"] / max(elapsed, 1e-6)
            print(
                f"[train] epoch={epoch} step={step}/{len(loader)} "
                f"loss={totals['total']/totals['count']:.4f} "
                f"global={totals['global']/totals['count']:.4f} "
                f"mask={totals['mask']/totals['count']:.4f} "
                f"ips={images_per_second:.2f}",
                flush=True,
            )
    count = max(totals["count"], 1)
    elapsed = time.perf_counter() - start_time
    return {
        "loss": totals["total"] / count,
        "loss_global": totals["global"] / count,
        "loss_mask": totals["mask"] / count,
        "loss_dice": totals["dice"] / count,
        "images_per_second": totals["count"] / max(elapsed, 1e-6),
    }


@torch.no_grad()
def evaluate_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    cfg: Config,
    concept_to_idx: Dict[str, int],
    n_concepts: int,
    split_name: str,
) -> Dict[str, float]:
    head.eval()
    totals = {"total": 0.0, "global": 0.0, "mask": 0.0, "dice": 0.0, "count": 0}
    start_time = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        images = prepare_images(batch["images"], cfg)
        global_targets, idx_pad, mask_pad, valid_pad = build_gdino_targets(
            batch["annotations"],
            batch["image_sizes"],
            concept_to_idx,
            n_concepts,
            cfg,
            cfg.device,
        )
        with autocast_context(cfg):
            feats = backbone(images)
            outputs = head(feats)
            losses = compute_losses(outputs, global_targets, idx_pad, mask_pad, valid_pad, cfg)
        batch_size = int(images.shape[0])
        totals["total"] += float(losses["total"].item()) * batch_size
        totals["global"] += float(losses["global"].item()) * batch_size
        totals["mask"] += float(losses["mask"].item()) * batch_size
        totals["dice"] += float(losses["dice"].item()) * batch_size
        totals["count"] += batch_size
        if step % cfg.log_every == 0:
            elapsed = time.perf_counter() - start_time
            images_per_second = totals["count"] / max(elapsed, 1e-6)
            print(
                f"[{split_name}] step={step}/{len(loader)} "
                f"loss={totals['total']/totals['count']:.4f} "
                f"ips={images_per_second:.2f}",
                flush=True,
            )
    count = max(totals["count"], 1)
    elapsed = time.perf_counter() - start_time
    return {
        "loss": totals["total"] / count,
        "loss_global": totals["global"] / count,
        "loss_mask": totals["mask"] / count,
        "loss_dice": totals["dice"] / count,
        "images_per_second": totals["count"] / max(elapsed, 1e-6),
    }


def build_run_dir(cfg: Config) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    name = cfg.run_name or f"savlg_imagenet_standalone_{timestamp}"
    run_dir = Path(cfg.save_dir) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_checkpoint(
    run_dir: Path,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Config,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        run_dir / f"checkpoint_epoch_{epoch:03d}.pt",
    )
    torch.save(head.state_dict(), run_dir / "concept_head_latest.pt")
    payload = {
        "epoch": epoch,
        "train": train_metrics,
        "val": val_metrics,
    }
    with (run_dir / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def profile_pipeline(cfg: Config) -> Dict[str, Any]:
    concepts = load_concepts(cfg.concept_file)
    dataset = SafeImageFolderWithAnnotations(
        root=cfg.train_root,
        annotation_dir=cfg.annotation_dir,
        concepts=concepts,
        input_size=cfg.input_size,
        min_image_bytes=cfg.min_image_bytes,
        split="train",
    )
    indices = list(range(len(dataset)))
    if cfg.max_train_images > 0:
        indices = indices[: cfg.max_train_images]
    loader = build_loader(DatasetView(dataset, indices), cfg, shuffle=False, drop_last=False)
    backbone, head = build_model(cfg, n_concepts=len(concepts))
    iterator = iter(loader)
    data_seconds = 0.0
    h2d_seconds = 0.0
    target_seconds = 0.0
    forward_seconds = 0.0
    profiled_steps = 0
    first_shapes: Optional[Dict[str, Any]] = None
    total_steps = cfg.warmup_steps + cfg.profile_steps
    for step in range(total_steps):
        t0 = time.perf_counter()
        batch = next(iterator)
        t1 = time.perf_counter()
        images = prepare_images(batch["images"], cfg)
        t2 = time.perf_counter()
        global_targets, idx_pad, mask_pad, valid_pad = build_gdino_targets(
            batch["annotations"],
            batch["image_sizes"],
            dataset.concept_to_idx,
            len(concepts),
            cfg,
            cfg.device,
        )
        t3 = time.perf_counter()
        with torch.no_grad():
            with autocast_context(cfg):
                feats = backbone(images)
                outputs = head(feats)
                _ = compute_losses(outputs, global_targets, idx_pad, mask_pad, valid_pad, cfg)
        if str(cfg.device).startswith("cuda"):
            torch.cuda.synchronize()
        t4 = time.perf_counter()
        if first_shapes is None:
            first_shapes = {
                "images": list(batch["images"].shape),
                "conv4": list(feats["conv4"].shape),
                "conv5": list(feats["conv5"].shape),
                "final_logits": list(outputs["final_logits"].shape),
                "idx_pad": list(idx_pad.shape),
                "mask_pad": list(mask_pad.shape),
            }
        if step >= cfg.warmup_steps:
            profiled_steps += 1
            data_seconds += t1 - t0
            h2d_seconds += t2 - t1
            target_seconds += t3 - t2
            forward_seconds += t4 - t3
    total_images = profiled_steps * cfg.batch_size
    total_seconds = data_seconds + h2d_seconds + target_seconds + forward_seconds
    return {
        "config": asdict(cfg),
        "n_concepts": len(concepts),
        "profiled_steps": profiled_steps,
        "images_profiled": total_images,
        "timing_seconds": {
            "data": data_seconds,
            "h2d": h2d_seconds,
            "targets": target_seconds,
            "forward_plus_loss": forward_seconds,
            "total": total_seconds,
        },
        "throughput": {
            "images_per_second_total": total_images / max(total_seconds, 1e-6),
            "images_per_second_compute": total_images / max(forward_seconds, 1e-6),
        },
        "shapes": first_shapes,
    }


def run_training(cfg: Config) -> Dict[str, Any]:
    concepts = load_concepts(cfg.concept_file)
    train_dataset_full = SafeImageFolderWithAnnotations(
        root=cfg.train_root,
        annotation_dir=cfg.annotation_dir,
        concepts=concepts,
        input_size=cfg.input_size,
        min_image_bytes=cfg.min_image_bytes,
        split="train",
    )
    if cfg.val_root:
        val_dataset_full = SafeImageFolderWithAnnotations(
            root=cfg.val_root,
            annotation_dir=cfg.annotation_dir,
            concepts=concepts,
            input_size=cfg.input_size,
            min_image_bytes=cfg.min_image_bytes,
            split="val",
        )
        train_indices = list(range(len(train_dataset_full)))
        if cfg.max_train_images > 0:
            train_indices = train_indices[: cfg.max_train_images]
        val_indices = list(range(len(val_dataset_full)))
        if cfg.max_val_images > 0:
            val_indices = val_indices[: cfg.max_val_images]
        train_dataset = DatasetView(train_dataset_full, train_indices)
        val_dataset = DatasetView(val_dataset_full, val_indices)
    else:
        train_dataset, val_dataset = split_train_val(
            train_dataset_full,
            val_split=cfg.val_split,
            max_train_images=cfg.max_train_images,
            max_val_images=cfg.max_val_images,
            seed=cfg.seed,
        )

    train_loader = build_loader(train_dataset, cfg, shuffle=True, drop_last=True)
    val_loader = build_loader(val_dataset, cfg, shuffle=False, drop_last=False)
    backbone, head = build_model(cfg, n_concepts=len(concepts))
    optimizer = make_optimizer(head, cfg)
    scaler = None
    if cfg.amp == "fp16" and str(cfg.device).startswith("cuda"):
        scaler = torch.cuda.amp.GradScaler()
    run_dir = build_run_dir(cfg)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run_dir / "concepts.txt").write_text("\n".join(concepts))

    best_val = float("inf")
    best_path = run_dir / "concept_head_best.pt"
    history: List[Dict[str, Any]] = []
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            backbone,
            head,
            train_loader,
            optimizer,
            scaler,
            cfg,
            concept_to_idx=train_dataset.concept_to_idx,
            n_concepts=len(concepts),
            epoch=epoch,
        )
        val_metrics = evaluate_one_epoch(
            backbone,
            head,
            val_loader,
            cfg,
            concept_to_idx=val_dataset.concept_to_idx,
            n_concepts=len(concepts),
            split_name="val",
        )
        summary = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(summary)
        print(
            f"[epoch] {epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_ips={train_metrics['images_per_second']:.2f} "
            f"val_loss={val_metrics['loss']:.4f} val_ips={val_metrics['images_per_second']:.2f}",
            flush=True,
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(head.state_dict(), best_path)
        if epoch % cfg.save_every == 0:
            save_checkpoint(run_dir, head, optimizer, epoch, cfg, train_metrics, val_metrics)

    result = {
        "run_dir": str(run_dir),
        "best_val_loss": best_val,
        "history": history,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "n_concepts": len(concepts),
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    cfg = build_config(parse_args())
    configure_runtime(cfg)
    if cfg.print_config:
        print(json.dumps(asdict(cfg), indent=2, sort_keys=True))
    if cfg.mode == "profile":
        result = profile_pipeline(cfg)
    else:
        result = run_training(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
