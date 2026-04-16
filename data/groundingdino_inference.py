"""GroundingDINO inference helpers used by the SAM3 concept-mask pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict


class Resize:
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, img, target):
        return T.RandomResize([self.size])(img, target)


def load_groundingdino_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    model.to(device)
    return model


class GroundingDinoBoxRunner:
    def __init__(self, config: Dict[str, Any]):
        self.device = str(config.get("device", "cuda"))
        self.box_threshold = float(config.get("box_threshold", 0.25))
        self.nms_iou_threshold = float(config.get("nms_iou_threshold", 0.5))
        resize = int(config.get("resize", 800))
        repo_path = str(config.get("repo_path", "GroundingDINO")).strip()
        if repo_path:
            repo = Path(repo_path).expanduser()
            if repo.exists() and str(repo) not in __import__("sys").path:
                __import__("sys").path.insert(0, str(repo))
        model_config = str(config.get("config") or "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py")
        checkpoint = str(config.get("checkpoint") or "GroundingDINO/groundingdino_swinb_cogcoor.pth")
        self.model = load_groundingdino_model(model_config, checkpoint, device=self.device)
        self.transform = T.Compose(
            [
                Resize(resize),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict_boxes(self, image_path: str, prompt: str, top_k: int = 5) -> List[Dict[str, Any]]:
        image = Image.open(image_path).convert("RGB")
        image_tensor, _ = self.transform(image, None)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        prompt = prompt.strip().rstrip(".") + " ."
        with torch.no_grad():
            outputs = self.model(image_tensor, captions=[prompt])
        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]
        scores = logits.max(dim=-1).values
        keep = scores > self.box_threshold
        if int(keep.sum().item()) <= 0:
            return []
        boxes = boxes[keep]
        scores = scores[keep]
        w, h = image.size
        cx, cy, bw, bh = boxes.unbind(-1)
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        keep_nms = nms(boxes_xyxy, scores, self.nms_iou_threshold)
        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(keep_nms[: int(top_k)].tolist()):
            results.append(
                {
                    "candidate_rank": int(rank),
                    "box_xyxy": [float(x) for x in boxes_xyxy[idx].tolist()],
                    "score": float(scores[idx].item()),
                    "source_index": int(idx),
                }
            )
        return results
