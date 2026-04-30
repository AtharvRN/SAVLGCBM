import argparse
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from transformers import ResNetForImageClassification

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import utils as data_utils
from scripts.eval_salf_imagenet_nec_tar import SoftmaxPooling2D
from scripts.eval_savlg_imagenet_standalone_val_tar import (
    load_run_config,
    resolve_final_layer_path,
    resolve_source_run_dir,
)
from scripts.train_savlg_imagenet_standalone import amp_dtype, build_model, configure_runtime, prepare_images


VAL_RE = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render top class-contributing concept maps for ImageNet CBM checkpoints."
    )
    parser.add_argument("--val_tar", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models", default="savlg,salf,vlg", help="Comma-separated subset of savlg,salf,vlg.")
    parser.add_argument("--savlg_artifact_dir", default="")
    parser.add_argument("--salf_dir", default="/root/salf-cbm_models/imagenet")
    parser.add_argument("--vlg_dir", default="/workspace/saved_models/imagenet_lf_cbm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=6)
    parser.add_argument("--start_image", type=int, default=1)
    parser.add_argument("--concepts_per_image", type=int, default=3)
    parser.add_argument("--page_size", type=int, default=6)
    parser.add_argument("--map_size", default="12,12")
    return parser.parse_args()


def parse_map_size(raw: str) -> Tuple[int, int]:
    vals = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(vals) != 2:
        raise ValueError("--map_size must be H,W")
    return vals[0], vals[1]


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


def overlay_heatmap(image_np: np.ndarray, heatmap: torch.Tensor) -> np.ndarray:
    heat = normalize_map(heatmap).cpu().numpy()
    rgba = plt.get_cmap("jet")(heat)[..., :3]
    return np.clip(0.58 * image_np + 0.42 * rgba, 0.0, 1.0)


def imagenet_crop_display(image: Image.Image, input_size: int = 224, resize_size: int = 256) -> np.ndarray:
    crop = transforms.Compose([transforms.Resize(resize_size), transforms.CenterCrop(input_size)])
    return np.asarray(crop(image)).astype(np.float32) / 255.0


def iter_val_images(val_tar: Path, start_image: int, max_images: int) -> Iterable[Tuple[int, str, Image.Image]]:
    used = 0
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
            used += 1
            if used >= max_images:
                break


def top_positive_contributions(
    concept_logits: torch.Tensor,
    final_weight: torch.Tensor,
    class_index: int,
    k: int,
) -> List[Tuple[int, float, float, float]]:
    contrib = concept_logits * final_weight[class_index]
    positive = contrib.clamp_min(0.0)
    if int((positive > 0).sum().item()) == 0:
        values, indices = contrib.abs().topk(k=min(k, contrib.numel()))
    else:
        values, indices = positive.topk(k=min(k, positive.numel()))
    out: List[Tuple[int, float, float, float]] = []
    for idx, value in zip(indices.tolist(), values.tolist()):
        out.append((int(idx), float(value), float(concept_logits[idx].item()), float(final_weight[class_index, idx].item())))
    return out


def load_linear_payload(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "weight" in payload and "bias" in payload:
        return payload["weight"].float(), payload["bias"].float()
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        return state["weight"].float(), state["bias"].float()
    raise ValueError(f"Could not read linear weight/bias from {path}")


class SAVLGRenderer:
    name = "SAVLG"

    def __init__(self, artifact_dir: Path, device: str) -> None:
        if not artifact_dir:
            raise ValueError("--savlg_artifact_dir is required for model savlg")
        self.artifact_dir = artifact_dir.resolve()
        self.source_run_dir = resolve_source_run_dir(self.artifact_dir)
        args_stub = argparse.Namespace(device=device, batch_size=1)
        self.cfg = load_run_config(self.source_run_dir, args_stub)
        self.cfg.batch_size = 1
        self.cfg.workers = 0
        configure_runtime(self.cfg)
        self.concepts = [line.strip() for line in (self.source_run_dir / "concepts.txt").read_text().splitlines() if line.strip()]
        self.backbone, self.head = build_model(self.cfg, n_concepts=len(self.concepts))
        self.head.load_state_dict(torch.load(self.source_run_dir / "concept_head_best.pt", map_location=self.cfg.device))
        self.backbone.eval()
        self.head.eval()
        weight, bias = load_linear_payload(resolve_final_layer_path(self.artifact_dir))
        norm = torch.load(self.artifact_dir / "final_layer_normalization.pt", map_location="cpu")
        self.mean = norm["mean"].to(self.cfg.device).float()
        self.std = norm["std"].to(self.cfg.device).float().clamp_min(1e-6)
        self.weight = weight.to(self.cfg.device).float()
        self.bias = bias.to(self.cfg.device).float()
        self.display_size = int(self.cfg.input_size)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.display_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def render_image(self, image: Image.Image, concepts_per_image: int) -> Dict[str, Any]:
        tensor = self.transform(image).unsqueeze(0)
        tensor = prepare_images(tensor, self.cfg)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype(self.cfg.amp),
                enabled=(str(self.cfg.device).startswith("cuda") and amp_dtype(self.cfg.amp) is not None),
            ):
                feats = self.backbone(tensor)
                outputs = self.head(feats)
                concept_logits = ((outputs["final_logits"].float().squeeze(0) - self.mean) / self.std).float()
                class_logits = F.linear(concept_logits.unsqueeze(0), self.weight, self.bias).squeeze(0)
        pred = int(class_logits.argmax().item())
        top = top_positive_contributions(concept_logits, self.weight, pred, concepts_per_image)
        maps = outputs["spatial_maps"].float().squeeze(0)
        image_np = imagenet_crop_display(image, self.display_size, 256)
        recs = []
        for concept_idx, contribution, activation, class_weight in top:
            heat = F.interpolate(
                maps[concept_idx].view(1, 1, *maps.shape[-2:]),
                size=image_np.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            recs.append(
                {
                    "concept_index": concept_idx,
                    "concept": self.concepts[concept_idx],
                    "contribution": contribution,
                    "activation": activation,
                    "class_weight": class_weight,
                    "heatmap": normalize_map(heat),
                    "overlay": overlay_heatmap(image_np, heat),
                }
            )
        return {"pred_class": pred, "pred_logit": float(class_logits[pred].item()), "image_np": image_np, "concepts": recs}


class SALFRenderer:
    name = "SALF-CBM"

    def __init__(self, load_dir: Path, device: str, map_size: Tuple[int, int]) -> None:
        self.load_dir = load_dir.resolve()
        self.device = device
        self.map_size = map_size
        self.concepts = canonicalize_concepts(self.load_dir / "concepts.txt")
        target_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(device).eval()
        self.backbone = torch.nn.Sequential(*list(target_model.resnet.children())[:-1]).to(device).eval()
        self.w_c = torch.load(self.load_dir / "W_c.pt", map_location=device).float()
        self.weight = torch.load(self.load_dir / "W_g.pt", map_location=device).float()
        self.bias = torch.load(self.load_dir / "b_g.pt", map_location=device).float()
        self.mean = torch.load(self.load_dir / "proj_mean.pt", map_location=device).float().flatten()
        self.std = torch.load(self.load_dir / "proj_std.pt", map_location=device).float().flatten().clamp_min(1e-6)
        self.pool = SoftmaxPooling2D(map_size).to(device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def render_image(self, image: Image.Image, concepts_per_image: int) -> Dict[str, Any]:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.backbone(tensor).last_hidden_state.float()
            if tuple(feats.shape[-2:]) != self.map_size:
                feats = F.interpolate(feats, size=self.map_size, mode="bilinear", align_corners=False)
            maps = F.conv2d(feats, self.w_c)
            pooled = self.pool(maps).flatten(1).squeeze(0)
            concept_logits = (pooled - self.mean) / self.std
            class_logits = F.linear(concept_logits.unsqueeze(0), self.weight, self.bias).squeeze(0)
            maps_norm = (maps.squeeze(0) - self.mean[:, None, None]) / self.std[:, None, None]
        pred = int(class_logits.argmax().item())
        top = top_positive_contributions(concept_logits, self.weight, pred, concepts_per_image)
        image_np = imagenet_crop_display(image, 224, 224)
        recs = []
        for concept_idx, contribution, activation, class_weight in top:
            heat = F.interpolate(
                maps_norm[concept_idx].view(1, 1, *maps_norm.shape[-2:]),
                size=image_np.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            recs.append(
                {
                    "concept_index": concept_idx,
                    "concept": self.concepts[concept_idx],
                    "contribution": contribution,
                    "activation": activation,
                    "class_weight": class_weight,
                    "heatmap": normalize_map(heat),
                    "overlay": overlay_heatmap(image_np, heat),
                }
            )
        return {"pred_class": pred, "pred_logit": float(class_logits[pred].item()), "image_np": image_np, "concepts": recs}


class VLGRenderer:
    name = "VLG-CBM"

    def __init__(self, load_dir: Path, device: str) -> None:
        self.load_dir = load_dir.resolve()
        self.device = device
        self.concepts = canonicalize_concepts(self.load_dir / "concepts.txt")
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        resnet = models.resnet50(weights=weights).to(device).eval()
        self.stem = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        ).to(device).eval()
        self.w_c = torch.load(self.load_dir / "W_c.pt", map_location=device).float()
        self.weight = torch.load(self.load_dir / "W_g.pt", map_location=device).float()
        self.bias = torch.load(self.load_dir / "b_g.pt", map_location=device).float()
        self.mean = torch.load(self.load_dir / "proj_mean.pt", map_location=device).float().flatten()
        self.std = torch.load(self.load_dir / "proj_std.pt", map_location=device).float().flatten().clamp_min(1e-6)
        self.transform = weights.transforms()

    def render_image(self, image: Image.Image, concepts_per_image: int) -> Dict[str, Any]:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.stem(tensor).float()
            maps = F.conv2d(feats, self.w_c[:, :, None, None])
            pooled = maps.mean(dim=(2, 3)).squeeze(0)
            concept_logits = (pooled - self.mean) / self.std
            class_logits = F.linear(concept_logits.unsqueeze(0), self.weight, self.bias).squeeze(0)
            maps_norm = (maps.squeeze(0) - self.mean[:, None, None]) / self.std[:, None, None]
        pred = int(class_logits.argmax().item())
        top = top_positive_contributions(concept_logits, self.weight, pred, concepts_per_image)
        image_np = imagenet_crop_display(image, 224, 256)
        recs = []
        for concept_idx, contribution, activation, class_weight in top:
            heat = F.interpolate(
                maps_norm[concept_idx].view(1, 1, *maps_norm.shape[-2:]),
                size=image_np.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            recs.append(
                {
                    "concept_index": concept_idx,
                    "concept": self.concepts[concept_idx],
                    "contribution": contribution,
                    "activation": activation,
                    "class_weight": class_weight,
                    "heatmap": normalize_map(heat),
                    "overlay": overlay_heatmap(image_np, heat),
                }
            )
        return {"pred_class": pred, "pred_logit": float(class_logits[pred].item()), "image_np": image_np, "concepts": recs}


def save_pages(records: Sequence[Dict[str, Any]], output_dir: Path, model_name: str, page_size: int) -> List[str]:
    page_paths: List[str] = []
    safe_name = model_name.lower().replace("-", "").replace(" ", "_")
    for page_idx, start in enumerate(range(0, len(records), page_size), start=1):
        page = records[start : start + page_size]
        fig, axes = plt.subplots(len(page), 4, figsize=(16, max(3.4 * len(page), 3.4)), squeeze=False)
        for row, item in enumerate(page):
            image_np = item["image_np"]
            concept_items = item["concepts"]
            axes[row, 0].imshow(image_np)
            axes[row, 0].set_title(f"{item['image_name']}\n{model_name} pred={item['pred_class']} logit={item['pred_logit']:.2f}")
            axes[row, 0].axis("off")
            for col in range(1, 4):
                ax = axes[row, col]
                concept_pos = col - 1
                if concept_pos >= len(concept_items):
                    ax.axis("off")
                    continue
                rec = concept_items[concept_pos]
                ax.imshow(rec["overlay"])
                ax.set_title(
                    f"#{concept_pos + 1}: {rec['concept']}\n"
                    f"contrib={rec['contribution']:.2f}, act={rec['activation']:.2f}, w={rec['class_weight']:.2f}",
                    fontsize=9,
                )
                ax.axis("off")
        fig.tight_layout()
        path = output_dir / f"{safe_name}_top_concepts_page_{page_idx:03d}.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        page_paths.append(str(path))
    return page_paths


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    images = list(iter_val_images(Path(args.val_tar).resolve(), args.start_image, args.max_images))
    summaries: Dict[str, Any] = {
        "val_tar": str(Path(args.val_tar).resolve()),
        "output_dir": str(output_dir),
        "images": [{"index": idx, "name": name} for idx, name, _ in images],
        "models": {},
    }

    renderers = []
    map_size = parse_map_size(args.map_size)
    for name in model_names:
        if name == "savlg":
            renderers.append(SAVLGRenderer(Path(args.savlg_artifact_dir), args.device))
        elif name == "salf":
            renderers.append(SALFRenderer(Path(args.salf_dir), args.device, map_size))
        elif name == "vlg":
            renderers.append(VLGRenderer(Path(args.vlg_dir), args.device))
        else:
            raise ValueError(f"Unknown model name: {name}")

    for renderer in renderers:
        records: List[Dict[str, Any]] = []
        json_records: List[Dict[str, Any]] = []
        for image_index, image_name, image in images:
            rendered = renderer.render_image(image, args.concepts_per_image)
            item = {
                "image_index": image_index,
                "image_name": image_name,
                **rendered,
            }
            records.append(item)
            json_records.append(
                {
                    "image_index": image_index,
                    "image_name": image_name,
                    "pred_class": rendered["pred_class"],
                    "pred_logit": rendered["pred_logit"],
                    "top_concepts": [
                        {k: v for k, v in rec.items() if k not in {"heatmap", "overlay"}}
                        for rec in rendered["concepts"]
                    ],
                }
            )
        pages = save_pages(records, output_dir, renderer.name, args.page_size)
        summaries["models"][renderer.name] = {"records": json_records, "pages": pages}

    summary_path = output_dir / "top_concept_render_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
