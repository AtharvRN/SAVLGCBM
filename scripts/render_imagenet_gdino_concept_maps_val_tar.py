import argparse
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import utils as data_utils
from model.cbm import Backbone, ConceptLayer
from scripts.imagenet_annotation_index import (
    build_filename_to_annotation_path,
    load_annotation_payload,
    resolve_val_annotation_dir,
)
from scripts.eval_savlg_imagenet_standalone_localization import normalize_maps
from scripts.eval_savlg_imagenet_standalone_val_tar import load_run_config, resolve_source_run_dir
from scripts.train_savlg_imagenet_standalone import amp_dtype, build_model, configure_runtime, prepare_images


VAL_RE = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG$")


class SoftmaxPooling2D(torch.nn.Module):
    def __init__(self, kernel_size: Tuple[int, int]) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, _h, _w = x.shape
        kh, kw = self.kernel_size
        patches = F.unfold(x, kernel_size=(kh, kw), stride=(kh, kw))
        patches = patches.view(n, c, kh * kw, -1)
        weights = F.softmax(patches, dim=2)
        return (patches * weights).sum(dim=2).view(n, c, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render GDINO concepts on ImageNet val with VLG Grad-CAM, SALF native maps, and SAVLG native maps."
    )
    parser.add_argument("--val_tar", required=True)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument(
        "--annotation_val_root",
        default="",
        help="Optional reorganized ImageNet val ImageFolder root used when annotations are keyed by ImageFolder dataset index.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", default="vlg,salf,savlg", help="Comma-separated subset of vlg,salf,savlg.")
    parser.add_argument("--vlg_load_dir", default="", help="VLG-CBM checkpoint dir with args.txt and cbl.pt.")
    parser.add_argument("--salf_dir", default="/root/salf-cbm_models/imagenet")
    parser.add_argument("--savlg_artifact_dir", default="")
    parser.add_argument("--start_image", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=6)
    parser.add_argument(
        "--image_names",
        default="",
        help="Optional comma-separated exact val filenames (e.g. ILSVRC2012_val_00041481.JPEG). If set, these are rendered exactly and start_image/max_images are ignored.",
    )
    parser.add_argument("--concepts_per_image", type=int, default=3)
    parser.add_argument("--page_concepts", type=int, default=3)
    parser.add_argument(
        "--map_normalization",
        default="concept_zscore_minmax",
        choices=["minmax", "sigmoid", "concept_zscore_minmax"],
        help="Normalization used for native concept maps before visualization. VLG Grad-CAM remains min-max normalized.",
    )
    parser.add_argument(
        "--select_by",
        default="max_pred",
        choices=["annotation_logit", "max_pred", "vlg_score", "salf_score", "savlg_score"],
        help="How to rank GDINO-present concepts before rendering.",
    )
    parser.add_argument("--target_concepts", default="", help="Optional comma-separated concept names to restrict rendering.")
    return parser.parse_args()


def canonicalize_concepts(path: Path) -> List[str]:
    return [
        data_utils.canonicalize_concept_label(line.strip())
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    x = x - x.min()
    return x / x.max().clamp_min(1e-6)


def normalize_map_with_mode(x: torch.Tensor, mode: str) -> torch.Tensor:
    x = x.detach().float()
    if x.ndim != 2:
        raise ValueError(f"Expected 2D map, got shape {tuple(x.shape)}")
    return normalize_maps(x.unsqueeze(0), mode).squeeze(0)


def overlay_heatmap(image_np: np.ndarray, heatmap: torch.Tensor) -> np.ndarray:
    heat = normalize_map(heatmap).cpu().numpy()
    rgba = plt.get_cmap("jet")(heat)[..., :3]
    return np.clip(0.58 * image_np + 0.42 * rgba, 0.0, 1.0)


def add_box(ax, box: Sequence[float], color: str, label: str = "") -> None:
    x1, y1, x2, y2 = [float(v) for v in box]
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2.0))
    if label:
        ax.text(
            x1,
            max(2.0, y1 - 4.0),
            label,
            color=color,
            fontsize=8,
            weight="bold",
            backgroundcolor=(1, 1, 1, 0.68),
        )


def union_boxes(boxes: Sequence[Sequence[float]]) -> Optional[List[float]]:
    valid = [box for box in boxes if isinstance(box, (list, tuple)) and len(box) == 4]
    if not valid:
        return None
    return [
        min(float(box[0]) for box in valid),
        min(float(box[1]) for box in valid),
        max(float(box[2]) for box in valid),
        max(float(box[3]) for box in valid),
    ]


def load_annotations(
    annotation_val_dir: Path,
    image_index_1based: int,
    image_name: str,
    concept_to_idx: Dict[str, int],
    filename_to_annotation_path: Optional[Dict[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    payload = load_annotation_payload(
        annotation_val_dir=annotation_val_dir,
        image_index_1based=image_index_1based,
        image_name=image_name,
        filename_to_annotation_path=filename_to_annotation_path,
    )
    entries = payload[1:] if isinstance(payload, list) else payload.get("concepts", [])
    grouped: Dict[str, Dict[str, Any]] = {}
    for ann in entries:
        if not isinstance(ann, dict):
            continue
        label = ann.get("label")
        if isinstance(label, str):
            label = data_utils.canonicalize_concept_label(label)
        if not isinstance(label, str) or label not in concept_to_idx:
            continue
        box = ann.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        rec = grouped.setdefault(label, {"boxes": [], "annotation_logit": float("-inf")})
        rec["boxes"].append([float(v) for v in box])
        rec["annotation_logit"] = max(float(rec["annotation_logit"]), float(ann.get("logit", 0.0)))
    return grouped


def iter_val_images(val_tar: Path, start_image: int, max_images: int) -> Iterable[Tuple[int, str, Image.Image]]:
    seen = 0
    with tarfile.open(val_tar, "r|*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            match = VAL_RE.search(Path(member.name).name)
            if match is None:
                continue
            image_index = int(match.group(1))
            if image_index < start_image:
                continue
            handle = tf.extractfile(member)
            if handle is None:
                raise FileNotFoundError(member.name)
            with Image.open(handle) as image:
                yield image_index, Path(member.name).name, image.convert("RGB")
            seen += 1
            if max_images > 0 and seen >= max_images:
                break


def iter_val_images_by_name(
    val_tar: Path,
    image_names: Sequence[str],
) -> Iterable[Tuple[int, str, Image.Image]]:
    requested = [name.strip() for name in image_names if name.strip()]
    remaining = set(requested)
    if not requested:
        return
    with tarfile.open(val_tar, "r|*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            image_name = Path(member.name).name
            if image_name not in remaining:
                continue
            match = VAL_RE.search(image_name)
            if match is None:
                continue
            image_index = int(match.group(1))
            handle = tf.extractfile(member)
            if handle is None:
                raise FileNotFoundError(member.name)
            with Image.open(handle) as image:
                yield image_index, image_name, image.convert("RGB")
            remaining.remove(image_name)
            if not remaining:
                break
    if remaining:
        missing = ", ".join(sorted(remaining))
        raise FileNotFoundError(f"Could not find requested images in tar: {missing}")


def image_to_input(image: Image.Image, input_size: int, resize_size: int) -> Tuple[np.ndarray, torch.Tensor]:
    transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    image_np = np.asarray(transforms.Compose([transforms.Resize(resize_size), transforms.CenterCrop(input_size)])(image)).astype(np.float32) / 255.0
    return image_np, transform(image)


def concept_contributions(concept_logits: torch.Tensor, class_weight: torch.Tensor, class_idx: int) -> torch.Tensor:
    return concept_logits * class_weight[class_idx]


def top_concepts_for_image(
    grouped: Dict[str, Dict[str, Any]],
    concept_to_idx: Dict[str, int],
    concept_limit: int,
    score_lookup: Dict[str, Dict[str, float]],
    select_by: str,
) -> List[Tuple[str, Dict[str, Any], int, float]]:
    ordered = list(grouped.items())
    def rank_value(item: Tuple[str, Dict[str, Any]]) -> float:
        concept, rec = item
        if select_by == "annotation_logit":
            return float(rec["annotation_logit"])
        scores = score_lookup.get(concept, {})
        if select_by == "max_pred":
            return max(scores.values()) if scores else float("-inf")
        model_name = {
            "vlg_score": "VLG-CBM",
            "salf_score": "SALF-CBM",
            "savlg_score": "SAVLG",
        }[select_by]
        return float(scores.get(model_name, float("-inf")))

    ordered.sort(key=rank_value, reverse=True)
    out: List[Tuple[str, Dict[str, Any], int, float]] = []
    for concept, rec in ordered:
        cidx = concept_to_idx[concept]
        out.append((concept, rec, cidx, rank_value((concept, rec))))
        if len(out) >= concept_limit:
            break
    return out


class VLGRenderer:
    def __init__(self, load_dir: Path, device: str) -> None:
        self.name = "VLG-CBM"
        self.load_dir = load_dir.resolve()
        self.device = device
        with (self.load_dir / "args.txt").open("r", encoding="utf-8") as handle:
            model_args = json.load(handle)
        self.backbone = Backbone(model_args["backbone"], model_args["feature_layer"], device).eval()
        self.concepts = canonicalize_concepts(self.load_dir / "concepts.txt")
        self.concept_to_idx = {concept: idx for idx, concept in enumerate(self.concepts)}
        self.use_cbl = (self.load_dir / "cbl.pt").is_file()
        if self.use_cbl:
            self.concept_layer = ConceptLayer.from_pretrained(str(self.load_dir), device=device).eval()
            self.w_c = None
            self.proj_mean = None
            self.proj_std = None
        else:
            self.concept_layer = None
            self.w_c = torch.load(self.load_dir / "W_c.pt", map_location=device).float()
            self.proj_mean = torch.load(self.load_dir / "proj_mean.pt", map_location=device).float().flatten()
            self.proj_std = torch.load(self.load_dir / "proj_std.pt", map_location=device).float().flatten().clamp_min(1e-6)
        self.input_size = 224
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def concept_map(self, image_tensor: torch.Tensor, concept_name: str) -> Tuple[torch.Tensor, float]:
        batch = image_tensor.unsqueeze(0).to(self.device)
        self.backbone.zero_grad(set_to_none=True)
        if self.concept_layer is not None:
            self.concept_layer.zero_grad(set_to_none=True)
        _ = self.backbone.backbone(batch)
        feats = self.backbone.feature_vals[batch.device]
        pooled = feats.mean(dim=(2, 3))
        concept_idx = self.concept_to_idx[concept_name]
        if self.use_cbl:
            logits = self.concept_layer(pooled).float().squeeze(0)
            target = logits[concept_idx]
            score = float(target.item())
        else:
            concept_logits = ((pooled @ self.w_c.T).squeeze(0) - self.proj_mean) / self.proj_std
            target = concept_logits[concept_idx]
            score = float(target.item())
        grad = torch.autograd.grad(target, feats, retain_graph=False, create_graph=False)[0]
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * feats).sum(dim=1, keepdim=False)).squeeze(0).detach()
        return cam, score

    def concept_score(self, image_tensor: torch.Tensor, concept_name: str) -> float:
        batch = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            _ = self.backbone.backbone(batch)
            feats = self.backbone.feature_vals[batch.device]
            pooled = feats.mean(dim=(2, 3)).squeeze(0)
            concept_idx = self.concept_to_idx[concept_name]
            if self.use_cbl:
                logits = self.concept_layer(pooled.unsqueeze(0)).float().squeeze(0)
                return float(logits[concept_idx].item())
            concept_logits = (pooled @ self.w_c.T - self.proj_mean) / self.proj_std
            return float(concept_logits[concept_idx].item())


class SAVLGRenderer:
    def __init__(self, artifact_dir: Path, device: str) -> None:
        self.name = "SAVLG"
        self.artifact_dir = artifact_dir.resolve()
        self.source_run_dir = resolve_source_run_dir(self.artifact_dir)
        args_stub = argparse.Namespace(device=device, batch_size=1)
        self.cfg = load_run_config(self.source_run_dir, args_stub)
        self.cfg.batch_size = 1
        self.cfg.workers = 0
        configure_runtime(self.cfg)
        self.concepts = [line.strip() for line in (self.source_run_dir / "concepts.txt").read_text().splitlines() if line.strip()]
        self.concept_to_idx = {concept: idx for idx, concept in enumerate(self.concepts)}
        self.backbone, self.head = build_model(self.cfg, n_concepts=len(self.concepts))
        self.head.load_state_dict(torch.load(self.source_run_dir / "concept_head_best.pt", map_location=self.cfg.device))
        self.backbone.eval()
        self.head.eval()
        norm = torch.load(self.artifact_dir / "final_layer_normalization.pt", map_location="cpu")
        linear_payload = torch.load(self.artifact_dir / "final_layer_dense.pt", map_location="cpu")
        self.mean = norm["mean"].to(self.cfg.device).float()
        self.std = norm["std"].to(self.cfg.device).float().clamp_min(1e-6)
        self.weight = linear_payload["weight"].to(self.cfg.device).float()
        self.bias = linear_payload["bias"].to(self.cfg.device).float()
        self.input_size = int(self.cfg.input_size)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def concept_map(self, image_tensor: torch.Tensor, concept_name: str) -> Tuple[torch.Tensor, float]:
        batch = prepare_images(image_tensor.unsqueeze(0), self.cfg)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype(self.cfg.amp),
                enabled=(str(self.cfg.device).startswith("cuda") and amp_dtype(self.cfg.amp) is not None),
            ):
                feats = self.backbone(batch)
                outputs = self.head(feats)
                concept_logits = ((outputs["final_logits"].float().squeeze(0) - self.mean) / self.std).float()
        concept_idx = self.concept_to_idx[concept_name]
        return outputs["spatial_maps"].float().squeeze(0)[concept_idx].detach(), float(concept_logits[concept_idx].item())

    def concept_score(self, image_tensor: torch.Tensor, concept_name: str) -> float:
        batch = prepare_images(image_tensor.unsqueeze(0), self.cfg)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype(self.cfg.amp),
                enabled=(str(self.cfg.device).startswith("cuda") and amp_dtype(self.cfg.amp) is not None),
            ):
                feats = self.backbone(batch)
                outputs = self.head(feats)
                concept_logits = ((outputs["final_logits"].float().squeeze(0) - self.mean) / self.std).float()
        return float(concept_logits[self.concept_to_idx[concept_name]].item())


class SALFRenderer:
    def __init__(self, load_dir: Path, device: str) -> None:
        self.name = "SALF-CBM"
        self.load_dir = load_dir.resolve()
        self.device = device
        self.concepts = canonicalize_concepts(self.load_dir / "concepts.txt")
        self.concept_to_idx = {concept: idx for idx, concept in enumerate(self.concepts)}
        target_model, _ = data_utils.get_target_model("resnet50", device)
        self.backbone = torch.nn.Sequential(*list(target_model.children())[:-2]).to(device).eval()
        self.w_c = torch.load(self.load_dir / "W_c.pt", map_location=device).float()
        self.w_c_conv = self.w_c[:, :, None, None]
        self.weight = torch.load(self.load_dir / "W_g.pt", map_location=device).float()
        self.bias = torch.load(self.load_dir / "b_g.pt", map_location=device).float()
        self.mean = torch.load(self.load_dir / "proj_mean.pt", map_location=device).float().flatten()
        self.std = torch.load(self.load_dir / "proj_std.pt", map_location=device).float().flatten().clamp_min(1e-6)
        self.pool = SoftmaxPooling2D((12, 12)).to(device)
        self.input_size = 224
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def concept_map(self, image_tensor: torch.Tensor, concept_name: str) -> Tuple[torch.Tensor, float]:
        batch = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.backbone(batch).float()
            if tuple(feats.shape[-2:]) != (12, 12):
                feats = F.interpolate(feats, size=(12, 12), mode="bilinear", align_corners=False)
            maps = F.conv2d(feats, self.w_c_conv)
            pooled = self.pool(maps).flatten(1).squeeze(0)
            concept_logits = (pooled - self.mean) / self.std
        concept_idx = self.concept_to_idx[concept_name]
        return maps.squeeze(0).float()[concept_idx].detach(), float(concept_logits[concept_idx].item())

    def concept_score(self, image_tensor: torch.Tensor, concept_name: str) -> float:
        batch = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.backbone(batch).float()
            if tuple(feats.shape[-2:]) != (12, 12):
                feats = F.interpolate(feats, size=(12, 12), mode="bilinear", align_corners=False)
            maps = F.conv2d(feats, self.w_c_conv)
            pooled = self.pool(maps).flatten(1).squeeze(0)
            concept_logits = (pooled - self.mean) / self.std
        return float(concept_logits[self.concept_to_idx[concept_name]].item())


def render_page(
    image_name: str,
    image_np: np.ndarray,
    concept_rows: Sequence[Tuple[str, Dict[str, Any], int, float, Dict[str, Any]]],
    output_path: Path,
) -> None:
    rows = len(concept_rows)
    fig, axes = plt.subplots(rows, 4, figsize=(18, max(4.0 * rows, 4.0)), squeeze=False)
    for row, (concept, rec, concept_idx, annotation_logit, model_outputs) in enumerate(concept_rows):
        gt_box = union_boxes(rec["boxes"])
        axes[row, 0].imshow(image_np)
        if gt_box is not None:
            add_box(axes[row, 0], gt_box, "#d62728", "GDINO")
        axes[row, 0].set_title(f"{image_name}\n{concept}\nann_logit={annotation_logit:.2f}")
        axes[row, 0].axis("off")
        model_titles = [("VLG-CBM", "#1f77b4"), ("SALF-CBM", "#2ca02c"), ("SAVLG", "#ff7f0e")]
        for col, (model_name, _color) in enumerate(model_titles, start=1):
            outputs = model_outputs[model_name]
            overlay = outputs["overlay"]
            axes[row, col].imshow(overlay)
            if gt_box is not None:
                add_box(axes[row, col], gt_box, "#d62728", "GDINO")
            axes[row, col].set_title(
                f"{model_name}\nlogit={outputs['score']:.3f}, disp={outputs['display_activation']:.3f}, raw={outputs['raw_activation']:.3f}"
            )
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    models_to_render = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_val_dir = resolve_val_annotation_dir(Path(args.annotation_dir).resolve())
    annotation_val_root = Path(args.annotation_val_root).resolve() if args.annotation_val_root else None
    filename_to_annotation_path = build_filename_to_annotation_path(annotation_val_dir, annotation_val_root)
    target_concepts = {
        data_utils.canonicalize_concept_label(item.strip())
        for item in args.target_concepts.split(",")
        if item.strip()
    }

    renderers = {}
    if "vlg" in models_to_render:
        if not args.vlg_load_dir:
            raise ValueError("--vlg_load_dir is required when rendering VLG-CBM")
        renderers["VLG-CBM"] = VLGRenderer(Path(args.vlg_load_dir), args.device)
    if "salf" in models_to_render:
        renderers["SALF-CBM"] = SALFRenderer(Path(args.salf_dir), args.device)
    if "savlg" in models_to_render:
        if not args.savlg_artifact_dir:
            raise ValueError("--savlg_artifact_dir is required when rendering SAVLG")
        renderers["SAVLG"] = SAVLGRenderer(Path(args.savlg_artifact_dir), args.device)

    concept_names = set()
    for renderer in renderers.values():
        concept_names.update(renderer.concepts)
    concept_to_idx = {name: idx for idx, name in enumerate(sorted(concept_names))}
    common_concepts = set.intersection(*(set(renderer.concepts) for renderer in renderers.values())) if renderers else set()
    summary: Dict[str, Any] = {
        "val_tar": str(Path(args.val_tar).resolve()),
        "annotation_dir": str(annotation_val_dir),
        "annotation_val_root": str(annotation_val_root) if annotation_val_root is not None else "",
        "output_dir": str(output_dir),
        "models": list(renderers.keys()),
        "images": [],
    }

    if args.image_names:
        image_iter = iter_val_images_by_name(
            Path(args.val_tar).resolve(),
            [item.strip() for item in args.image_names.split(",") if item.strip()],
        )
    else:
        image_iter = iter_val_images(Path(args.val_tar).resolve(), args.start_image, args.max_images)

    for image_index, image_name, image in image_iter:
        grouped = load_annotations(
            annotation_val_dir,
            image_index,
            image_name,
            concept_to_idx,
            filename_to_annotation_path=filename_to_annotation_path,
        )
        if target_concepts:
            grouped = {concept: rec for concept, rec in grouped.items() if concept in target_concepts}
        if common_concepts:
            grouped = {concept: rec for concept, rec in grouped.items() if concept in common_concepts}
        if not grouped:
            continue

        image_np = np.asarray(transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(image)).astype(np.float32) / 255.0
        image_tensor_cache: Dict[str, torch.Tensor] = {}
        score_lookup: Dict[str, Dict[str, float]] = {}

        for model_name, renderer in renderers.items():
            if model_name not in image_tensor_cache:
                image_tensor_cache[model_name] = renderer.transform(image)
            for concept in grouped:
                score_lookup.setdefault(concept, {})[model_name] = renderer.concept_score(image_tensor_cache[model_name], concept)

        row_data = top_concepts_for_image(
            grouped,
            concept_to_idx,
            args.concepts_per_image,
            score_lookup,
            args.select_by,
        )

        concept_rows = []
        for concept, rec, concept_idx, _dummy_score in row_data:
            concept_outputs: Dict[str, Dict[str, Any]] = {}
            for model_name, renderer in renderers.items():
                if model_name not in image_tensor_cache:
                    tensor = renderer.transform(image)
                    image_tensor_cache[model_name] = tensor
                heat_raw, score = renderer.concept_map(image_tensor_cache[model_name], concept)
                heat = F.interpolate(
                    heat_raw.view(1, 1, *heat_raw.shape[-2:]),
                    size=image_np.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                if model_name == "VLG-CBM":
                    display_heat = normalize_map(heat)
                else:
                    display_heat = normalize_map_with_mode(heat, args.map_normalization)
                concept_outputs[model_name] = {
                    "overlay": overlay_heatmap(image_np, display_heat),
                    "score": score,
                    "raw_activation": float(heat_raw.max().item()),
                    "display_activation": float(display_heat.max().item()),
                }
            concept_rows.append((concept, rec, concept_idx, float(rec["annotation_logit"]), concept_outputs))

        page_path = output_dir / f"{Path(image_name).stem}_concepts.png"
        render_page(image_name, image_np, concept_rows[: args.page_concepts], page_path)
        summary["images"].append(
            {
                "image_index": image_index,
                "image_name": image_name,
                "page": str(page_path),
                "concepts": [
                    {
                        "concept": concept,
                        "rank_score": float(rank_score),
                        "model_scores": score_lookup.get(concept, {}),
                        "annotation_logit": float(rec["annotation_logit"]),
                        "boxes": rec["boxes"],
                    }
                    for concept, rec, _, rank_score, _ in concept_rows[: args.page_concepts]
                ],
            }
        )

    (output_dir / "gdino_concept_render_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
