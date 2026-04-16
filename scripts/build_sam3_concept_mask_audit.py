#!/usr/bin/env python3
"""Build a lightweight HTML audit page for a SAM3 concept-mask cache."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.sam3_concept_mask_cache import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an HTML audit page for SAM3 preview overlays")
    parser.add_argument("--manifest", required=True, help="Path to a SAM3 concept-mask manifest.json")
    parser.add_argument("--output", default=None, help="Output HTML path. Defaults to <split>/audit/index.html")
    parser.add_argument("--max_records", type=int, default=300, help="Maximum records to include")
    parser.add_argument("--include_errors", action="store_true", help="Include non-ok records")
    return parser.parse_args()


def _rel_url(path: Path, output_dir: Path) -> str:
    rel = os.path.relpath(path.resolve(), output_dir.resolve())
    return quote(str(rel), safe="/._-:")


def _record_sort_key(record: Dict[str, Any]) -> tuple:
    return (
        int(record.get("dataset_index", -1)),
        int(record.get("concept_index", -1)),
        str(record.get("concept", "")),
    )


def _read_records(manifest: Dict[str, Any], include_errors: bool, max_records: int) -> List[Dict[str, Any]]:
    records = sorted(manifest.get("records", []), key=_record_sort_key)
    if not include_errors:
        records = [record for record in records if record.get("status") == "ok"]
    return records[: int(max_records)]


def build_html(manifest_path: Path, output_path: Path, include_errors: bool, max_records: int) -> str:
    manifest = load_manifest(manifest_path)
    manifest_root = manifest_path.resolve().parent
    output_dir = output_path.resolve().parent
    records = _read_records(manifest, include_errors=include_errors, max_records=max_records)

    status_counts: Dict[str, int] = {}
    for record in manifest.get("records", []):
        status = str(record.get("status", "missing"))
        status_counts[status] = status_counts.get(status, 0) + 1

    cards = []
    for record in records:
        preview_path = manifest_root / str(record.get("preview_path", ""))
        mask_path = manifest_root / str(record.get("mask_path", ""))
        image_path = str(record.get("image_path", ""))
        score = record.get("score")
        score_text = "n/a" if score is None else f"{float(score):.3f}"
        det = record.get("num_detections", "n/a")
        area_ratio = record.get("area_ratio")
        area_text = "n/a" if area_ratio is None else f"{float(area_ratio):.4f}"
        selection_mode = str(record.get("mask_selection", "n/a"))
        selection_fallback_from = record.get("selection_fallback_from")
        preview_html = (
            f'<img src="{_rel_url(preview_path, output_dir)}" alt="preview">'
            if preview_path.exists()
            else '<div class="missing">missing preview</div>'
        )
        candidate_items = []
        for candidate in record.get("candidate_summaries", []):
            candidate_preview_path = manifest_root / str(candidate.get("candidate_preview_path", ""))
            candidate_selected = bool(candidate.get("selected"))
            candidate_classes = "candidate selected" if candidate_selected else "candidate"
            candidate_img = (
                f'<img src="{_rel_url(candidate_preview_path, output_dir)}" alt="candidate">'
                if candidate_preview_path.exists()
                else '<div class="thumb-missing">missing</div>'
            )
            candidate_items.append(
                "\n".join(
                    [
                        f'<a class="{candidate_classes}" href="{_rel_url(candidate_preview_path, output_dir)}">',
                        candidate_img,
                        '<div class="candidate-meta">',
                        f'<span>r{int(candidate.get("candidate_rank", -1))} | src {int(candidate.get("source_index", -1))}</span>',
                        f'<span>score {float(candidate.get("score", 0.0)):.3f} | area {float(candidate.get("area_ratio", 0.0)):.4f}</span>',
                        f'<span>metric {float(candidate.get("selection_metric", 0.0)):.4f}</span>',
                        '</div>',
                        '</a>',
                    ]
                )
            )
        fallback_text = f" | fallback from: {selection_fallback_from}" if selection_fallback_from else ""
        cards.append(
            "\n".join(
                [
                    '<article class="card">',
                    f'<a href="{_rel_url(preview_path, output_dir)}">{preview_html}</a>',
                    '<div class="meta">',
                    f'<strong>{html.escape(str(record.get("concept", "")))}</strong>',
                    f'<span>image {int(record.get("dataset_index", -1)):06d} | concept {int(record.get("concept_index", -1)):04d}</span>',
                    f'<span>status: {html.escape(str(record.get("status", "")))} | score: {score_text} | area: {area_text} | detections: {html.escape(str(det))}</span>',
                    f'<span>selection: {html.escape(selection_mode)}{html.escape(fallback_text)}</span>',
                    f'<span>mask: {html.escape(str(mask_path.relative_to(manifest_root))) if mask_path.exists() else "missing"}</span>',
                    f'<span class="path">{html.escape(image_path)}</span>',
                    '</div>',
                    f'<div class="candidate-grid">{"".join(candidate_items) if candidate_items else ""}</div>',
                    '</article>',
                ]
            )
        )

    summary = {
        "manifest": str(manifest_path),
        "dataset": manifest.get("dataset"),
        "split": manifest.get("split"),
        "image_count": manifest.get("image_count"),
        "concept_count": manifest.get("concept_count"),
        "selected_concept_indices": manifest.get("selected_concept_indices"),
        "record_count": manifest.get("record_count"),
        "status_counts": status_counts,
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SAM3 Concept Mask Audit</title>
  <style>
    body {{ margin: 24px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; background: #f6f7f9; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    pre {{ padding: 12px; background: #111827; color: #f9fafb; overflow: auto; border-radius: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; margin-top: 20px; }}
    .card {{ background: white; border: 1px solid #d7dce2; border-radius: 6px; overflow: hidden; }}
    .card img {{ display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #0b1020; }}
    .missing {{ display: grid; place-items: center; height: 190px; color: #9b1c1c; background: #fee2e2; }}
    .meta {{ display: grid; gap: 4px; padding: 10px; font-size: 13px; line-height: 1.35; }}
    .meta span {{ color: #52616f; }}
    .path {{ overflow-wrap: anywhere; font-size: 11px; }}
    .candidate-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(104px, 1fr)); gap: 8px; padding: 0 10px 10px; }}
    .candidate {{ display: grid; gap: 4px; color: inherit; text-decoration: none; }}
    .candidate.selected {{ outline: 2px solid #0f766e; }}
    .candidate img {{ aspect-ratio: 1 / 1; border: 1px solid #d7dce2; background: #0b1020; }}
    .candidate-meta {{ display: grid; gap: 2px; font-size: 11px; color: #52616f; }}
    .thumb-missing {{ display: grid; place-items: center; aspect-ratio: 1 / 1; font-size: 11px; color: #9b1c1c; background: #fee2e2; border: 1px solid #fecaca; }}
  </style>
</head>
<body>
  <h1>SAM3 Concept Mask Audit</h1>
  <pre>{html.escape(json.dumps(summary, indent=2))}</pre>
  <section class="grid">
    {"".join(cards)}
  </section>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_path = Path(args.output) if args.output else manifest_path.parent / "audit" / "index.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_html(
            manifest_path=manifest_path,
            output_path=output_path,
            include_errors=bool(args.include_errors),
            max_records=int(args.max_records),
        )
    )
    print(json.dumps({"audit_html": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
