import argparse
import csv
import json
import os
import textwrap
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

import data.utils as data_utils
from methods.common import load_run_info
from scripts.evaluate_concept_interventions import (
    _build_test_loader,
    _get_batch_model_state,
    _load_checkpoint_args,
    _load_concepts,
    _resolve_num_images,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute class-wise concept contributions on CUB and export Sankey-ready links."
        )
    )
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--class_source",
        type=str,
        default="gt",
        choices=["gt", "pred"],
        help="Whether to aggregate contributions toward the ground-truth class or predicted class.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="positive",
        choices=["positive", "signed", "abs"],
        help="How to aggregate per-concept class contributions.",
    )
    parser.add_argument(
        "--topk_per_class",
        type=int,
        default=10,
        help="Number of concepts to keep per class in Sankey exports.",
    )
    parser.add_argument(
        "--savlg_score_source",
        type=str,
        default="final",
        choices=["global", "spatial", "final"],
        help="Concept score source for SAVLG checkpoints.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def _accumulate_mode(contrib: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "positive":
        return torch.clamp(contrib, min=0.0)
    if mode == "signed":
        return contrib
    if mode == "abs":
        return contrib.abs()
    raise ValueError(mode)


def _safe_slug(raw: str) -> str:
    out = []
    for ch in raw.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "/"}:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "class"


def _write_sankey_html(output_path: Path, links: List[Dict[str, object]]) -> None:
    node_names: List[str] = []
    node_to_idx: Dict[str, int] = {}

    def add_node(name: str) -> int:
        if name not in node_to_idx:
            node_to_idx[name] = len(node_names)
            node_names.append(name)
        return node_to_idx[name]

    source_idxs = []
    target_idxs = []
    values = []
    labels = []
    for rec in links:
        src = str(rec["source"])
        dst = str(rec["target"])
        val = float(rec["value"])
        source_idxs.append(add_node(src))
        target_idxs.append(add_node(dst))
        values.append(val)
        labels.append(f"{src} -> {dst}: {val:.6f}")

    html = f"""\
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Class-Concept Contribution Sankey</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
      background: #ffffff;
    }}
    #chart {{
      width: 100vw;
      height: 100vh;
    }}
  </style>
</head>
<body>
  <div id="chart"></div>
  <script>
    const data = [{{
      type: "sankey",
      orientation: "h",
      arrangement: "snap",
      node: {{
        pad: 12,
        thickness: 12,
        line: {{ color: "rgba(80,80,80,0.35)", width: 0.5 }},
        label: {json.dumps(node_names)},
      }},
      link: {{
        source: {json.dumps(source_idxs)},
        target: {json.dumps(target_idxs)},
        value: {json.dumps(values)},
        label: {json.dumps(labels)},
      }},
    }}];
    const layout = {{
      title: "Concept contributions to class logits",
      font: {{ size: 12 }},
      margin: {{ l: 20, r: 20, t: 50, b: 20 }},
    }};
    Plotly.newPlot("chart", data, layout, {{displayModeBar: true, responsive: true}});
  </script>
</body>
</html>
"""
    output_path.write_text(textwrap.dedent(html))


def main() -> None:
    os.chdir(Path(__file__).resolve().parents[1])
    cli_args = _parse_args()
    load_path = cli_args.load_path
    ckpt_args = _load_checkpoint_args(load_path, cli_args.device, cli_args.annotation_dir)
    concepts = _load_concepts(load_path)
    num_images = _resolve_num_images(cli_args)
    run_info = load_run_info(load_path)
    model_name = run_info.get("model_name", run_info["args"]["model_name"])

    if model_name == "savlg_cbm":
        setattr(ckpt_args, "savlg_score_source", cli_args.savlg_score_source)

    loader = _build_test_loader(
        model_name=model_name,
        load_path=load_path,
        args=ckpt_args,
        concepts=concepts,
        batch_size_override=cli_args.batch_size,
        num_workers_override=cli_args.num_workers,
        num_images=num_images,
    )

    class_names = data_utils.get_classes(ckpt_args.dataset)

    final_layer = None
    per_class_sum = torch.zeros(len(class_names), len(concepts), dtype=torch.float64)
    per_class_count = torch.zeros(len(class_names), dtype=torch.int64)

    logger.info(
        "Running contribution export: model={} images={} class_source={} mode={} topk={}",
        model_name,
        len(loader.dataset),
        cli_args.class_source,
        cli_args.mode,
        cli_args.topk_per_class,
    )

    for images, _concept_gt, targets in tqdm(loader, desc="Class contributions"):
        images = images.to(ckpt_args.device, non_blocking=True)
        targets = targets.to(ckpt_args.device, non_blocking=True)

        with torch.no_grad():
            concept_space, logits, _gt_transform = _get_batch_model_state(
                model_name=model_name,
                load_path=load_path,
                args=ckpt_args,
                images=images,
            )

        if final_layer is None:
            from scripts.evaluate_concept_interventions import (
                _lf_state,
                _salf_state,
                _savlg_state,
                _vlg_state,
            )

            if model_name == "lf_cbm":
                final_layer = _lf_state(load_path, ckpt_args).final
            elif model_name == "vlg_cbm":
                final_layer = _vlg_state(load_path, ckpt_args)[3]
            elif model_name == "salf_cbm":
                final_layer = _salf_state(load_path, ckpt_args)[4]
            elif model_name == "savlg_cbm":
                final_layer = _savlg_state(load_path, ckpt_args)[4]
            else:
                raise NotImplementedError(model_name)
            final_layer.eval()

        if cli_args.class_source == "gt":
            class_indices = targets
        else:
            class_indices = logits.argmax(dim=1)

        weights = final_layer.weight[class_indices]
        contrib = weights * concept_space
        contrib = _accumulate_mode(contrib, cli_args.mode).detach().cpu().to(torch.float64)
        class_indices_cpu = class_indices.detach().cpu()

        for row_idx, class_idx in enumerate(class_indices_cpu.tolist()):
            per_class_sum[class_idx] += contrib[row_idx]
            per_class_count[class_idx] += 1

    per_class_mean = torch.zeros_like(per_class_sum)
    valid_mask = per_class_count > 0
    per_class_mean[valid_mask] = per_class_sum[valid_mask] / per_class_count[valid_mask].unsqueeze(1)

    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    sankey_rows: List[Dict[str, object]] = []
    sankeymatic_lines: List[str] = []

    for class_idx, class_name in enumerate(class_names):
        if per_class_count[class_idx].item() == 0:
            continue
        values = per_class_mean[class_idx]
        topk = min(cli_args.topk_per_class, values.numel())
        top_vals, top_idxs = values.topk(k=topk)

        class_rows = []
        for rank, (concept_idx, value) in enumerate(zip(top_idxs.tolist(), top_vals.tolist()), start=1):
            row = {
                "class_idx": class_idx,
                "class_name": class_name,
                "rank": rank,
                "concept_idx": concept_idx,
                "concept_name": concepts[concept_idx],
                "mean_contribution": float(value),
                "num_examples": int(per_class_count[class_idx].item()),
            }
            class_rows.append(row)
            summary_rows.append(row)
            sankey_rows.append(
                {
                    "source": concepts[concept_idx],
                    "target": class_name,
                    "value": float(value),
                }
            )
            sankeymatic_lines.append(
                f"{concepts[concept_idx]} [{float(value):.6f}] {class_name}"
            )

        class_csv = output_dir / f"class_{class_idx:03d}_{_safe_slug(class_name)}.csv"
        pd.DataFrame(class_rows).to_csv(class_csv, index=False)

    summary_csv = output_dir / "class_contributions_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    sankey_csv = output_dir / "sankey_links.csv"
    with open(sankey_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "value"])
        writer.writeheader()
        writer.writerows(sankey_rows)

    sankeymatic_txt = output_dir / "sankeymatic_flows.txt"
    sankeymatic_txt.write_text("\n".join(sankeymatic_lines) + "\n")
    sankey_html = output_dir / "sankey.html"
    _write_sankey_html(sankey_html, sankey_rows)

    metadata = {
        "load_path": load_path,
        "model_name": model_name,
        "num_images": len(loader.dataset),
        "class_source": cli_args.class_source,
        "mode": cli_args.mode,
        "topk_per_class": cli_args.topk_per_class,
        "savlg_score_source": cli_args.savlg_score_source if model_name == "savlg_cbm" else None,
        "summary_csv": str(summary_csv),
        "sankey_csv": str(sankey_csv),
        "sankeymatic_txt": str(sankeymatic_txt),
        "sankey_html": str(sankey_html),
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Saved contribution exports to {}", output_dir)


if __name__ == "__main__":
    main()
