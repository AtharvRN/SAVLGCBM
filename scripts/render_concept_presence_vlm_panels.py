#!/usr/bin/env python3
"""Render per-image concept presence panels with VLM judgments for multiple CBMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image


JUDGE_COLORS = {
    "yes": "#7da87b",
    "no": "#c97c73",
    "unsure": "#c9b56b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        action="append",
        dest="model_dirs",
        required=True,
        help="Concept presence export directory. Pass once per model.",
    )
    parser.add_argument(
        "--judge-jsonl-name",
        type=str,
        default="judge_results_openai_common10images_b20.jsonl",
        help="Judge JSONL filename inside each model dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for rendered figures.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on number of shared images to render.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="Concept Presence",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _display_name(model_name: str) -> str:
    mapping = {
        "lf_cbm": "LF-CBM",
        "vlg_cbm": "VLG-CBM",
        "salf_cbm": "SALF-CBM",
        "savlg_cbm": "SAVLG-CBM",
    }
    return mapping.get(model_name, model_name)


def _collect_model_payload(model_dir: Path, judge_jsonl_name: str) -> dict[str, Any]:
    manifest = _load_json(model_dir / "manifest.json")
    judge_rows = _load_jsonl(model_dir / judge_jsonl_name)
    judge_by_task = {row["task_id"]: row for row in judge_rows}
    images_by_index = {int(image["dataset_index"]): image for image in manifest["images"]}
    model_name = manifest["metadata"]["model_name"]
    return {
        "dir": model_dir,
        "manifest": manifest,
        "judge_by_task": judge_by_task,
        "images_by_index": images_by_index,
        "model_name": model_name,
        "display_name": _display_name(model_name),
    }


def _concept_task_id(image: dict[str, Any], concept: dict[str, Any]) -> str:
    overlay_file = concept.get("overlay_file")
    if overlay_file:
        return Path(overlay_file).stem
    stem = (
        f"image_{image['subset_rank']:04d}_idx_{image['dataset_index']}"
        f"_concept_{concept['rank_in_image']:02d}_{concept['concept_index']:03d}"
        f"_{_slugify(concept['concept_name'])}"
    )
    return stem


def _slugify(raw: str) -> str:
    chars: list[str] = []
    for ch in raw.lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in {" ", "-", "_", "/"}:
            chars.append("_")
    return "".join(chars).strip("_")


def _shared_dataset_indices(model_payloads: list[dict[str, Any]]) -> list[int]:
    shared = set(model_payloads[0]["images_by_index"])
    for payload in model_payloads[1:]:
        shared &= set(payload["images_by_index"])
    return sorted(shared)


def _short_label(text: str, limit: int = 34) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _render_single_image(
    dataset_index: int,
    model_payloads: list[dict[str, Any]],
    output_dir: Path,
    title_prefix: str,
) -> None:
    ref_image = model_payloads[0]["images_by_index"][dataset_index]
    ref_dir = model_payloads[0]["dir"]
    image_path = ref_dir / ref_image["image_file"]
    image = Image.open(image_path).convert("RGB")

    fig = plt.figure(figsize=(18, 4.5))
    outer = gridspec.GridSpec(1, 5, width_ratios=[1.3, 1, 1, 1, 1], wspace=0.28)

    ax_img = fig.add_subplot(outer[0, 0])
    ax_img.imshow(image)
    ax_img.axis("off")
    ax_img.set_title(
        f"Image\nidx={dataset_index}\nGT={ref_image['target_class_name']}\nPred={ref_image['pred_class_name']}",
        fontsize=11,
    )

    for col, payload in enumerate(model_payloads, start=1):
        ax = fig.add_subplot(outer[0, col])
        image_record = payload["images_by_index"][dataset_index]
        concepts = image_record["concepts"]
        labels = []
        contributions = []
        annots = []
        colors = []
        for concept in concepts:
            task_id = _concept_task_id(image_record, concept)
            judge_row = payload["judge_by_task"].get(task_id)
            judge = judge_row["judge"]["concept_present"] if judge_row else "missing"
            conf = judge_row["judge"].get("concept_present_confidence") if judge_row else None
            labels.append(_short_label(concept["concept_name"]))
            contributions.append(float(concept["contribution_to_reference_class"]))
            activation = float(concept["activation_score"])
            if conf is None:
                annots.append(f"{activation:.2f} | {judge}")
            else:
                annots.append(f"{activation:.2f} | {judge} ({conf:.2f})")
            colors.append(JUDGE_COLORS.get(judge, "#9aa0a6"))

        y = list(range(len(concepts)))
        ax.barh(y, contributions, color=colors, alpha=0.95)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Contribution", fontsize=10)
        ax.set_title(payload["display_name"], fontsize=12)
        xmax = max(contributions) if contributions else 1.0
        ax.set_xlim(0, xmax * 1.28 if xmax > 0 else 1.0)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        for yi, value, text in zip(y, contributions, annots):
            ax.text(value + max(xmax * 0.02, 0.01), yi, text, va="center", fontsize=8)

    fig.suptitle(
        f"{title_prefix} | dataset_idx={dataset_index} | top contributing concepts with VLM presence judgment",
        fontsize=18,
        y=1.03,
    )
    fig.tight_layout()
    out_path = output_dir / f"concept_presence_panel_idx_{dataset_index}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_payloads = [
        _collect_model_payload(Path(model_dir), args.judge_jsonl_name)
        for model_dir in args.model_dirs
    ]
    shared_indices = _shared_dataset_indices(model_payloads)
    if args.max_images is not None:
        shared_indices = shared_indices[: args.max_images]

    for dataset_index in shared_indices:
        _render_single_image(
            dataset_index=dataset_index,
            model_payloads=model_payloads,
            output_dir=output_dir,
            title_prefix=args.title_prefix,
        )

    print(
        f"Rendered {len(shared_indices)} figures to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
