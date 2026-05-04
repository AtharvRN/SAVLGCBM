#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stanford_cars_common import (  # noqa: E402
    annotation_file_path,
    canonicalize_concept_label,
    load_jsonl,
    read_concepts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GroundingDINO concept-box annotations for Stanford Cars.")
    parser.add_argument("--manifest_dir", default="data/stanford_cars")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated splits from manifest_dir.")
    parser.add_argument("--concept_file", default="concept_files/stanford_cars_concepts_filtered.txt")
    parser.add_argument("--output_dir", default="annotations/stanford_cars")
    parser.add_argument("--config", default="GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py")
    parser.add_argument("--grounded_checkpoint", default="GroundingDINO/groundingdino_swinb_cogcoor.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prompt_chunk_size", type=int, default=24)
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--max_boxes_per_concept", type=int, default=2)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_images", type=int, default=0)
    return parser.parse_args()


def import_grounding_dino():
    try:
        import GroundingDINO.groundingdino.datasets.transforms as T
        from GroundingDINO.groundingdino.models import build_model
        from GroundingDINO.groundingdino.util.slconfig import SLConfig
        from GroundingDINO.groundingdino.util.utils import clean_state_dict
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "GroundingDINO is not importable. Ensure the dependency is installed or vendored "
            "before running Stanford Cars annotation generation."
        ) from exc
    return T, build_model, SLConfig, clean_state_dict


class ManifestImageDataset(Dataset):
    def __init__(self, manifest_path: Path, transform: Any) -> None:
        self.rows = load_jsonl(manifest_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        tensor = self.transform(image, None)[0]
        return tensor, row


def collate_rows(batch: Sequence[Tuple[torch.Tensor, Dict[str, Any]]]):
    images, rows = zip(*batch)
    return torch.stack(images, dim=0), list(rows)


def load_annotation_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    T, build_model, SLConfig, clean_state_dict = import_grounding_dino()
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    model.to(device)
    return model, model.tokenizer, T


def split_prompt(prompt: str, tokenizer: Any, phrases: Sequence[str]) -> List[Tuple[str, Sequence[int]]]:
    token_ids = tokenizer(prompt)["input_ids"][1:-1]
    split_token_idxs = [idx for idx, token_id in enumerate(token_ids) if token_id == 1012]
    spans: List[Tuple[str, Sequence[int]]] = []
    start = 0
    for phrase_idx, split_idx in enumerate(split_token_idxs):
        if phrase_idx >= len(phrases):
            break
        spans.append((phrases[phrase_idx], token_ids[start:split_idx]))
        start = split_idx + 1
    if len(spans) != len(phrases):
        raise RuntimeError(
            f"Prompt tokenization mismatch: expected {len(phrases)} phrases but recovered {len(spans)} from tokenizer output"
        )
    return spans


def build_prompt_chunks(concepts: Sequence[str], chunk_size: int, tokenizer: Any) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for start in range(0, len(concepts), chunk_size):
        phrases = list(concepts[start : start + chunk_size])
        prompt = " . ".join(phrases) + " ."
        spans = split_prompt(prompt, tokenizer, phrases)
        chunks.append({"prompt": prompt, "phrases": phrases, "spans": spans})
    return chunks


def cxcywh_to_xyxy(box: Sequence[float], image_size: Tuple[int, int]) -> List[float]:
    width, height = image_size
    cx, cy, bw, bh = [float(value) for value in box]
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    x1 = max(0.0, min(x1, width))
    x2 = max(0.0, min(x2, width))
    y1 = max(0.0, min(y1, height))
    y2 = max(0.0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return []
    return [x1, y1, x2, y2]


def concept_score_from_span(prompt_logits: np.ndarray, span: Sequence[int]) -> float:
    if len(span) == 0:
        return 0.0
    token_scores = prompt_logits[: len(span)]
    if token_scores.size == 0:
        return 0.0
    token_scores = np.clip(token_scores, 1e-6, 1.0)
    return float(np.exp(np.log(token_scores).mean()))


def process_chunk_predictions(
    *,
    rows: Sequence[Dict[str, Any]],
    logits: torch.Tensor,
    boxes: torch.Tensor,
    chunk: Dict[str, Any],
    box_threshold: float,
    text_threshold: float,
) -> List[List[Dict[str, Any]]]:
    batch_annotations: List[List[Dict[str, Any]]] = [[] for _ in rows]
    span_lengths = [len(span_tokens) for _phrase, span_tokens in chunk["spans"]]
    token_offsets = np.cumsum([0] + span_lengths)

    logits_np = logits.detach().cpu().numpy()
    boxes_np = boxes.detach().cpu().numpy()
    for batch_index, row in enumerate(rows):
        image_size = (int(row["original_width"]), int(row["original_height"]))
        for query_logits, query_box in zip(logits_np[batch_index], boxes_np[batch_index]):
            query_logits = query_logits[1:-1]
            if float(np.max(query_logits)) < float(box_threshold):
                continue
            box_xyxy = cxcywh_to_xyxy(query_box, image_size)
            if not box_xyxy:
                continue
            for phrase_index, phrase in enumerate(chunk["phrases"]):
                start = int(token_offsets[phrase_index])
                end = int(token_offsets[phrase_index + 1])
                score = concept_score_from_span(query_logits[start:end], range(end - start))
                if score < float(text_threshold):
                    continue
                batch_annotations[batch_index].append(
                    {
                        "label": phrase,
                        "prompt_phrase": phrase,
                        "canonical_label": canonicalize_concept_label(phrase),
                        "box_xyxy": box_xyxy,
                        "score": float(score),
                        "logit": float(score),
                        "source": "groundingdino",
                    }
                )
    return batch_annotations


def prune_boxes(entries: Sequence[Dict[str, Any]], max_boxes_per_concept: int) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["canonical_label"])].append(dict(entry))
    kept: List[Dict[str, Any]] = []
    for canonical_label in sorted(grouped):
        items = sorted(grouped[canonical_label], key=lambda item: float(item["score"]), reverse=True)
        kept.extend(items[: max(1, int(max_boxes_per_concept))])
    return kept


def iter_selected_rows(rows: Sequence[Dict[str, Any]], *, num_shards: int, shard_id: int, max_images: int) -> Iterable[Tuple[int, Dict[str, Any]]]:
    emitted = 0
    for index, row in enumerate(rows):
        if index % max(1, int(num_shards)) != int(shard_id):
            continue
        yield index, row
        emitted += 1
        if max_images > 0 and emitted >= max_images:
            break


def manifest_path_for_split(manifest_dir: Path, split: str) -> Path:
    path = manifest_dir / f"{split}_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest for split={split}: {path}")
    return path


def build_index_files(output_dir: Path, splits: Sequence[str]) -> Dict[str, Any]:
    filename_to_annotation: Dict[str, str] = {}
    split_summary: Dict[str, Any] = {}
    for split in splits:
        split_dir = output_dir / split
        if not split_dir.is_dir():
            continue
        image_files = sorted(split_dir.glob("*.json"))
        concept_total = 0
        for path in image_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            image_id = str(payload.get("image_id", path.stem))
            filename_to_annotation[image_id] = str(path)
            image_path = payload.get("image_path")
            if image_path:
                filename_to_annotation[Path(str(image_path)).name] = str(path)
            concept_total += len(payload.get("concepts", []))
        split_summary[split] = {
            "n_images": len(image_files),
            "n_concept_boxes": concept_total,
        }
    (output_dir / "filename_to_annotation.json").write_text(json.dumps(filename_to_annotation, indent=2), encoding="utf-8")
    summary = {
        "annotation_root": str(output_dir),
        "splits": split_summary,
    }
    (output_dir / "annotation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    manifest_dir = Path(args.manifest_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [token.strip() for token in args.splits.split(",") if token.strip()]
    if not splits:
        raise ValueError("No splits requested")
    if int(args.shard_id) < 0 or int(args.shard_id) >= max(1, int(args.num_shards)):
        raise ValueError("--shard_id must be in [0, num_shards)")

    concepts = read_concepts(Path(args.concept_file))
    if not concepts:
        raise ValueError(f"No concepts found in {args.concept_file}")

    model, tokenizer, T = load_annotation_model(
        model_config_path=args.config,
        model_checkpoint_path=args.grounded_checkpoint,
        device=args.device,
    )
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    prompt_chunks = build_prompt_chunks(concepts, max(1, int(args.prompt_chunk_size)), tokenizer)

    split_results: Dict[str, Any] = {}
    for split in splits:
        manifest_path = manifest_path_for_split(manifest_dir, split)
        dataset = ManifestImageDataset(manifest_path, transform=transform)
        selected_rows = list(iter_selected_rows(dataset.rows, num_shards=args.num_shards, shard_id=args.shard_id, max_images=args.max_images))
        selected_indices = [index for index, _row in selected_rows]
        if not selected_indices:
            split_results[split] = {"n_selected": 0, "n_written": 0, "n_skipped_existing": 0}
            continue
        subset = torch.utils.data.Subset(dataset, selected_indices)
        loader = DataLoader(
            subset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=str(args.device).startswith("cuda"),
            collate_fn=collate_rows,
        )

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        n_written = 0
        n_skipped_existing = 0
        for images, rows in tqdm(loader, desc=f"gdino:{split}"):
            pending_rows: List[Dict[str, Any]] = []
            pending_images: List[torch.Tensor] = []
            pending_output_paths: List[Path] = []
            for row, image in zip(rows, images):
                output_path = annotation_file_path(output_dir, split, str(row["image_id"]))
                if args.resume and output_path.is_file():
                    n_skipped_existing += 1
                    continue
                pending_rows.append(row)
                pending_images.append(image)
                pending_output_paths.append(output_path)
            if not pending_rows:
                continue

            image_tensor = torch.stack(pending_images, dim=0).to(args.device)
            aggregated: List[List[Dict[str, Any]]] = [[] for _ in pending_rows]
            with torch.no_grad():
                for chunk in prompt_chunks:
                    outputs = model(image_tensor, captions=[chunk["prompt"]] * image_tensor.shape[0])
                    chunk_annotations = process_chunk_predictions(
                        rows=pending_rows,
                        logits=outputs["pred_logits"].sigmoid(),
                        boxes=outputs["pred_boxes"],
                        chunk=chunk,
                        box_threshold=float(args.box_threshold),
                        text_threshold=float(args.text_threshold),
                    )
                    for batch_index, items in enumerate(chunk_annotations):
                        aggregated[batch_index].extend(items)

            for row, entries, output_path in zip(pending_rows, aggregated, pending_output_paths):
                payload = {
                    "image_path": str(row["image_path"]),
                    "image_id": str(row["image_id"]),
                    "class_id": int(row["class_id"]),
                    "class_name": str(row["class_name"]),
                    "split": split,
                    "original_size": [int(row["original_width"]), int(row["original_height"])],
                    "concepts": prune_boxes(entries, int(args.max_boxes_per_concept)),
                }
                output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                n_written += 1
        split_results[split] = {
            "n_selected": len(selected_indices),
            "n_written": n_written,
            "n_skipped_existing": n_skipped_existing,
        }

    annotation_summary = build_index_files(output_dir, splits)
    result = {
        "manifest_dir": str(manifest_dir),
        "output_dir": str(output_dir),
        "splits": split_results,
        "annotation_summary": annotation_summary,
        "n_concepts": len(concepts),
        "prompt_chunks": len(prompt_chunks),
        "box_threshold": float(args.box_threshold),
        "text_threshold": float(args.text_threshold),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
