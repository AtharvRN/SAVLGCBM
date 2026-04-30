#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize standalone SAVLG ImageNet run timings and telemetry.")
    parser.add_argument("--run_dir", required=True, help="Path to standalone run directory.")
    parser.add_argument(
        "--staging_json",
        default="",
        help="Optional path to stage-data timing JSON written by the full ImageNet job.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summarize(values: Iterable[float]) -> Dict[str, float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {}
    def pct(p: float) -> float:
        idx = min(len(vals) - 1, int(p * (len(vals) - 1)))
        return vals[idx]
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "p50": pct(0.5),
        "p90": pct(0.9),
        "max": vals[-1],
    }


def _read_telemetry(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metrics = {
        "gpu_util": "gpu_util",
        "gpu_mem_used_mb": "gpu_mem_used_mb",
        "gpu_power_w": "gpu_power_w",
        "proc_cpu_pct": "proc_cpu_pct",
        "proc_rss_mb": "proc_rss_mb",
        "mem_used_mb": "mem_used_mb",
        "cgroup_memory_current_mb": "cgroup_memory_current_mb",
    }
    payload: Dict[str, Dict[str, float]] = {}
    for out_key, in_key in metrics.items():
        vals: List[float] = []
        for row in rows:
            raw = row.get(in_key, "")
            if raw in ("", "nan", None):
                continue
            try:
                vals.append(float(raw))
            except ValueError:
                continue
        if vals:
            payload[out_key] = _summarize(vals)
    return payload


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary = _load_json(run_dir / "summary.json")
    result: Dict[str, object] = {
        "run_dir": str(run_dir),
        "train_size": summary.get("train_size"),
        "val_size": summary.get("val_size"),
        "n_concepts": summary.get("n_concepts"),
        "best_val_loss": summary.get("best_val_loss"),
    }

    history = summary.get("history") or []
    if history:
        result["history_last"] = history[-1]

    final_layer = summary.get("final_layer")
    if final_layer:
        result["final_layer"] = final_layer

    telemetry = _read_telemetry(run_dir / "system_telemetry.csv")
    if telemetry:
        result["system_telemetry"] = telemetry

    if args.staging_json:
        staging_path = Path(args.staging_json)
        if staging_path.exists():
            result["staging"] = _load_json(staging_path)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
