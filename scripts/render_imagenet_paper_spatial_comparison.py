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
from scipy.io import loadmat
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
from scripts.eval_savlg_imagenet_standalone_val_tar import (
    load_run_config,
    resolve_final_layer_path,
    resolve_source_run_dir,
)
from scripts.train_savlg_imagenet_standalone import amp_dtype, build_model, configure_runtime, prepare_images


VAL_RE = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG$")

DEFAULT_GROUP_PATTERNS: Dict[str, List[str]] = {
    "dog": ["dog", "retriever", "terrier", "spaniel", "poodle", "shepherd", "husky", "beagle", "hound"],
    "car": ["car", "cab", "limousine", "jeep", "convertible", "sports car", "racer", "minivan"],
    "bird": ["bird", "partridge", "jay", "magpie", "kite", "eagle", "owl", "sparrow", "finch", "hen", "cock", "duck", "goose"],
    "food": ["pizza", "hotdog", "cheeseburger", "bagel", "pretzel", "trifle", "ice cream", "guacamole"],
    "furniture": ["chair", "sofa", "couch", "table", "lamp", "bookcase", "wardrobe", "cabinet", "desk", "bench"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper-style ImageNet SAVLG vs VLG spatial comparisons.")
    parser.add_argument("--val_tar", required=True)
    parser.add_argument("--devkit_dir", required=True)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--annotation_val_root", default="")
    parser.add_argument("--savlg_artifact_dir", required=True)
    parser.add_argument("--vlg_load_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_names", default="")
    parser.add_argument("--groups", default="dog,car,bird,food,furniture")
    parser.add_argument("--concepts_per_image", type=int, default=3)
    parser.add_argument("--concept_manifest_json", default="")
    parser.add_argument("--map_normalization", default="concept_zscore_minmax", choices=["minmax", "sigmoid", "concept_zscore_minmax"])
    parser.add_argument("--max_scan_images", type=int, default=50000)
    return parser.parse_args()


def canonicalize_concepts(path: Path) -> List[str]:
    return [data_utils.canonicalize_concept_label(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    x = x - x.min()
    return x / x.max().clamp_min(1e-6)


def normalize_map_with_mode(x: torch.Tensor, mode: str) -> torch.Tensor:
    return normalize_maps(x.detach().float().unsqueeze(0), mode).squeeze(0)


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


def load_label_words(devkit_dir: Path) -> Dict[int, str]:
    payload = loadmat(devkit_dir / "data" / "meta.mat", squeeze_me=True, struct_as_record=False)
    synsets = payload["synsets"]
    id_to_wnid: Dict[int, str] = {}
    wnid_to_words: Dict[str, str] = {}
    for syn in synsets:
        ilsvrc_id = int(syn.ILSVRC2012_ID)
        if 1 <= ilsvrc_id <= 1000 and int(syn.num_children) == 0:
            wnid = str(syn.WNID)
            id_to_wnid[ilsvrc_id] = wnid
            wnid_to_words[wnid] = str(syn.words)
    wnids = sorted(id_to_wnid.values())
    class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
    idx_to_words = {class_to_idx[wnid]: wnid_to_words[wnid] for wnid in wnids}
    labels: List[int] = []
    with (devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                labels.append(class_to_idx[id_to_wnid[int(line)]])
    return {idx + 1: idx_to_words[label] for idx, label in enumerate(labels)}


def load_annotations(
    annotation_val_dir: Path,
    image_index_1based: int,
    image_name: str,
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
        if not isinstance(label, str):
            continue
        box = ann.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        rec = grouped.setdefault(label, {"boxes": [], "annotation_logit": float("-inf")})
        rec["boxes"].append([float(v) for v in box])
        rec["annotation_logit"] = max(float(rec["annotation_logit"]), float(ann.get("logit", 0.0)))
    return grouped


def iter_val_images(val_tar: Path, image_names: Sequence[str]) -> Iterable[Tuple[int, str, Image.Image]]:
    requested = [name.strip() for name in image_names if name.strip()]
    remaining = set(requested)
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
        raise FileNotFoundError(f"Could not find requested images: {sorted(remaining)}")


class VLGRenderer:
    def __init__(self, load_dir: Path, device: str) -> None:
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


class SAVLGRenderer:
    def __init__(self, artifact_dir: Path, device: str) -> None:
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
        weight_payload = torch.load(resolve_final_layer_path(self.artifact_dir), map_location="cpu")
        norm = torch.load(self.artifact_dir / "final_layer_normalization.pt", map_location="cpu")
        self.mean = norm["mean"].to(self.cfg.device).float()
        self.std = norm["std"].to(self.cfg.device).float().clamp_min(1e-6)
        self.weight = weight_payload["weight"].to(self.cfg.device).float()
        self.bias = weight_payload["bias"].to(self.cfg.device).float()
        self.display_size = int(self.cfg.input_size)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.display_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def image_outputs(self, image: Image.Image) -> Dict[str, Any]:
        tensor = self.transform(image).unsqueeze(0)
        batch = prepare_images(tensor, self.cfg)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype(self.cfg.amp),
                enabled=(str(self.cfg.device).startswith("cuda") and amp_dtype(self.cfg.amp) is not None),
            ):
                feats = self.backbone(batch)
                outputs = self.head(feats)
                concept_logits = ((outputs["final_logits"].float().squeeze(0) - self.mean) / self.std).float()
                class_logits = F.linear(concept_logits.unsqueeze(0), self.weight, self.bias).squeeze(0)
        pred = int(class_logits.argmax().item())
        return {
            "tensor": tensor.squeeze(0),
            "pred_class": pred,
            "pred_logit": float(class_logits[pred].item()),
            "class_logits": class_logits.detach(),
            "concept_logits": concept_logits.detach(),
            "spatial_maps": outputs["spatial_maps"].float().squeeze(0).detach(),
        }


def select_top_concepts(
    grouped: Dict[str, Dict[str, Any]],
    concept_to_idx: Dict[str, int],
    common_concepts: Sequence[str],
    concept_logits: torch.Tensor,
    weight: torch.Tensor,
    pred_class: int,
    top_k: int,
) -> List[Tuple[str, Dict[str, Any], int, float]]:
    rows: List[Tuple[str, Dict[str, Any], int, float]] = []
    fallback_rows: List[Tuple[str, Dict[str, Any], int, float]] = []
    for concept in common_concepts:
        if concept not in grouped:
            continue
        idx = concept_to_idx[concept]
        contrib = float((concept_logits[idx] * weight[pred_class, idx]).item())
        fallback_rows.append((concept, grouped[concept], idx, contrib))
        if contrib <= 0:
            continue
        rows.append((concept, grouped[concept], idx, contrib))
    rows.sort(key=lambda item: item[3], reverse=True)
    if rows:
        return rows[:top_k]
    fallback_rows.sort(key=lambda item: item[3], reverse=True)
    return fallback_rows[:top_k]


def render_figure(
    output_path: Path,
    image_name: str,
    class_label: str,
    image_np: np.ndarray,
    concept_rows: Sequence[Tuple[str, Dict[str, Any], int, float, Dict[str, Any]]],
) -> None:
    rows = len(concept_rows)
    fig, axes = plt.subplots(rows, 3, figsize=(14, max(3.8 * rows, 4.0)), squeeze=False)
    for row, (concept, rec, _concept_idx, contribution, model_outputs) in enumerate(concept_rows):
        gt_box = union_boxes(rec["boxes"])
        axes[row, 0].imshow(image_np)
        if gt_box is not None:
            add_box(axes[row, 0], gt_box, "#d62728", "GDINO")
        axes[row, 0].set_title(f"{image_name}\n{class_label}\n{concept}")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(model_outputs["savlg_overlay"])
        if gt_box is not None:
            add_box(axes[row, 1], gt_box, "#d62728", "GDINO")
        axes[row, 1].set_title(
            f"SAVLG native\ncontrib={contribution:.2f}, logit={model_outputs['savlg_score']:.2f}",
            fontsize=10,
        )
        axes[row, 1].axis("off")
        axes[row, 2].imshow(model_outputs["vlg_overlay"])
        if gt_box is not None:
            add_box(axes[row, 2], gt_box, "#d62728", "GDINO")
        axes[row, 2].set_title(f"VLG Grad-CAM\nlogit={model_outputs['vlg_score']:.2f}", fontsize=10)
        axes[row, 2].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def choose_images(
    args: argparse.Namespace,
    annotation_val_dir: Path,
    filename_to_annotation_path: Optional[Dict[str, Path]],
    label_words: Dict[int, str],
    common_concepts: Sequence[str],
) -> List[Tuple[str, str]]:
    if args.image_names:
        names = [item.strip() for item in args.image_names.split(",") if item.strip()]
        out = []
        for name in names:
            match = VAL_RE.search(name)
            if match is None:
                raise ValueError(f"Bad image name: {name}")
            idx = int(match.group(1))
            out.append((name, label_words[idx]))
        return out

    groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    found: Dict[str, Tuple[str, str]] = {}
    patterns = {group: DEFAULT_GROUP_PATTERNS[group] for group in groups if group in DEFAULT_GROUP_PATTERNS}
    with tarfile.open(Path(args.val_tar).resolve(), "r|*") as tf:
        for member in tf:
            if len(found) == len(patterns):
                break
            if not member.isfile():
                continue
            image_name = Path(member.name).name
            match = VAL_RE.search(image_name)
            if match is None:
                continue
            image_index = int(match.group(1))
            if image_index > args.max_scan_images:
                break
            label = label_words[image_index]
            grouped = load_annotations(
                annotation_val_dir,
                image_index,
                image_name,
                filename_to_annotation_path=filename_to_annotation_path,
            )
            if len(set(grouped).intersection(common_concepts)) < args.concepts_per_image:
                continue
            label_lower = label.lower()
            for group, pats in patterns.items():
                if group in found:
                    continue
                if any(p in label_lower for p in pats):
                    found[group] = (image_name, label)
                    break
    return [found[group] for group in groups if group in found]


def load_concept_manifest(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict) and "images" in raw:
        raw = raw["images"]
    manifest: Dict[str, Dict[str, Any]] = {}
    for image_name, payload in raw.items():
        if isinstance(payload, list):
            manifest[image_name] = {"concepts": payload}
        elif isinstance(payload, dict):
            manifest[image_name] = payload
        else:
            raise ValueError(f"Bad manifest entry for {image_name}: {type(payload)}")
    return manifest


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_val_dir = resolve_val_annotation_dir(Path(args.annotation_dir).resolve())
    annotation_val_root = Path(args.annotation_val_root).resolve() if args.annotation_val_root else None
    filename_to_annotation_path = build_filename_to_annotation_path(annotation_val_dir, annotation_val_root)
    label_words = load_label_words(Path(args.devkit_dir).resolve())

    savlg = SAVLGRenderer(Path(args.savlg_artifact_dir), args.device)
    vlg = VLGRenderer(Path(args.vlg_load_dir), args.device)
    common_concepts = sorted(set(savlg.concepts).intersection(vlg.concepts))
    concept_manifest = load_concept_manifest(args.concept_manifest_json)

    selected = choose_images(args, annotation_val_dir, filename_to_annotation_path, label_words, common_concepts)
    selected_names = [name for name, _label in selected]
    label_lookup = {name: label for name, label in selected}
    for image_name, payload in concept_manifest.items():
        if "label" in payload:
            label_lookup[image_name] = str(payload["label"])

    summary: Dict[str, Any] = {
        "val_tar": str(Path(args.val_tar).resolve()),
        "devkit_dir": str(Path(args.devkit_dir).resolve()),
        "annotation_dir": str(annotation_val_dir),
        "savlg_artifact_dir": str(Path(args.savlg_artifact_dir).resolve()),
        "vlg_load_dir": str(Path(args.vlg_load_dir).resolve()),
        "output_dir": str(output_dir),
        "images": [],
    }

    for image_index, image_name, image in iter_val_images(Path(args.val_tar).resolve(), selected_names):
        grouped = load_annotations(
            annotation_val_dir,
            image_index,
            image_name,
            filename_to_annotation_path=filename_to_annotation_path,
        )
        grouped = {k: v for k, v in grouped.items() if k in common_concepts}
        if not grouped:
            continue

        savlg_out = savlg.image_outputs(image)
        manifest_payload = concept_manifest.get(image_name)
        if manifest_payload and manifest_payload.get("concepts"):
            top_rows = []
            for concept in manifest_payload["concepts"]:
                canon = data_utils.canonicalize_concept_label(str(concept))
                if canon not in grouped or canon not in savlg.concept_to_idx:
                    continue
                concept_idx = savlg.concept_to_idx[canon]
                contribution = float((savlg_out["concept_logits"][concept_idx] * savlg.weight[savlg_out["pred_class"], concept_idx]).item())
                top_rows.append((canon, grouped[canon], concept_idx, contribution))
        else:
            top_rows = select_top_concepts(
                grouped,
                savlg.concept_to_idx,
                common_concepts,
                savlg_out["concept_logits"],
                savlg.weight,
                savlg_out["pred_class"],
                args.concepts_per_image,
            )
        if not top_rows:
            continue
        image_np = np.asarray(transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(image)).astype(np.float32) / 255.0
        concept_rows = []
        for concept, rec, concept_idx, contribution in top_rows:
            savlg_heat = F.interpolate(
                savlg_out["spatial_maps"][concept_idx].view(1, 1, *savlg_out["spatial_maps"].shape[-2:]),
                size=image_np.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            savlg_display = normalize_map_with_mode(savlg_heat, args.map_normalization)
            vlg_tensor = vlg.transform(image)
            vlg_cam, vlg_score = vlg.concept_map(vlg_tensor, concept)
            vlg_heat = F.interpolate(vlg_cam.view(1, 1, *vlg_cam.shape[-2:]), size=image_np.shape[:2], mode="bilinear", align_corners=False).squeeze()
            model_outputs = {
                "savlg_overlay": overlay_heatmap(image_np, savlg_display),
                "savlg_score": float(savlg_out["concept_logits"][concept_idx].item()),
                "vlg_overlay": overlay_heatmap(image_np, normalize_map(vlg_heat)),
                "vlg_score": float(vlg_score),
            }
            concept_rows.append((concept, rec, concept_idx, contribution, model_outputs))

        output_path = output_dir / f"{Path(image_name).stem}_savlg_vs_vlg.png"
        render_figure(output_path, image_name, label_lookup.get(image_name, label_words[image_index]), image_np, concept_rows)
        summary["images"].append(
            {
                "image_index": image_index,
                "image_name": image_name,
                "label": label_lookup.get(image_name, label_words[image_index]),
                "output_path": str(output_path),
                "concepts": [
                    {
                        "concept": concept,
                        "contribution": float(contribution),
                        "savlg_logit": float(model_outputs["savlg_score"]),
                        "vlg_logit": float(model_outputs["vlg_score"]),
                        "annotation_logit": float(rec["annotation_logit"]),
                        "boxes": rec["boxes"],
                    }
                    for concept, rec, _concept_idx, contribution, model_outputs in concept_rows
                ],
            }
        )

    (output_dir / "paper_spatial_comparison_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
