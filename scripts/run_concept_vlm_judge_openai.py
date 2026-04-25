#!/usr/bin/env python3
"""Run an OpenAI multimodal judge over exported concept-review tasks.

This script consumes `judge_tasks.jsonl` emitted by
`scripts/export_concept_judge_subset.py`, calls an OpenAI vision-language model
on each (image, overlay, concept) tuple, and writes structured judgments to a
resumable JSONL artifact plus a compact summary JSON.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field


SYSTEM_PROMPT = """You are a careful multimodal judge for a concept-bottleneck interpretability study.

You will receive:
- the original image
- a model heatmap overlay for one candidate concept
- a task-specific prompt describing the concept and the judging rules

Your job is to return a structured judgment only. Be conservative and use
"unsure" when the evidence is weak.
"""


class JudgeDecision(BaseModel):
    concept_present: str = Field(pattern="^(yes|no|unsure)$")
    concept_present_confidence: float = Field(ge=0.0, le=1.0)
    region_matches_concept: str = Field(pattern="^(yes|partial|no|unsure)$")
    region_matches_concept_confidence: float = Field(ge=0.0, le=1.0)
    rationale_short: str


class ConceptPresenceDecision(BaseModel):
    concept_present: str = Field(pattern="^(yes|no|unsure)$")
    concept_present_confidence: float = Field(ge=0.0, le=1.0)
    rationale_short: str


class SpatialFaithfulnessDecision(BaseModel):
    region_matches_concept: str = Field(pattern="^(yes|partial|no|unsure)$")
    region_matches_concept_confidence: float = Field(ge=0.0, le=1.0)
    rationale_short: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-jsonl",
        type=Path,
        required=True,
        help="Path to judge_tasks.jsonl from export_concept_judge_subset.py",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Where to write structured judge results. Defaults next to tasks file.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Where to write aggregate summary JSON. Defaults next to tasks file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4",
        help="OpenAI model name. Defaults to gpt-5.4.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional cap for debugging or staged runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing output JSONL and re-run all tasks.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional pause between requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per task before failing hard.",
    )
    parser.add_argument(
        "--task-id-prefix",
        type=str,
        default=None,
        help="Optional filter to run only task IDs with this prefix.",
    )
    return parser.parse_args()


def _default_output_paths(tasks_jsonl: Path) -> tuple[Path, Path]:
    root = tasks_jsonl.parent
    return root / "judge_results_openai.jsonl", root / "judge_results_openai_summary.json"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_existing(output_jsonl: Path, overwrite: bool) -> dict[str, dict[str, Any]]:
    if overwrite or not output_jsonl.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(output_jsonl):
        task_id = row.get("task_id")
        if task_id:
            existing[task_id] = row
    return existing


def _image_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        mime = "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{payload}"


def _resolve_path(base_dir: Path, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _call_openai(
    client: OpenAI,
    model: str,
    task: dict[str, Any],
    base_dir: Path,
) -> tuple[Any, Any]:
    image_path = _resolve_path(base_dir, task["image_file"])
    overlay_file = task.get("overlay_file")
    overlay_path = _resolve_path(base_dir, overlay_file) if overlay_file else None
    prompt = task["prompt_template"].strip()
    task_type = _infer_task_type(task)
    text_format = _text_format_for_task_type(task_type)
    user_content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_text", "text": "Original image:"},
        {
            "type": "input_image",
            "image_url": _image_to_data_url(image_path),
        },
    ]
    if overlay_path is not None:
        user_content.extend(
            [
                {"type": "input_text", "text": "Model heatmap overlay for the same image-concept pair:"},
                {
                    "type": "input_image",
                    "image_url": _image_to_data_url(overlay_path),
                },
            ]
        )
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_content,
            },
        ],
        text_format=text_format,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError(f"OpenAI response for task_id={task['task_id']} did not produce parsed output.")
    return parsed, response


def _infer_task_type(task: dict[str, Any]) -> str:
    task_type = task.get("task_type")
    if task_type:
        return str(task_type)
    schema = task.get("expected_response_schema") or {}
    required = set(schema.get("required", []))
    if "concept_present" in required and "region_matches_concept" in required:
        return "legacy_joint"
    if "region_matches_concept" in required:
        return "spatial_faithfulness"
    if "concept_present" in required:
        return "concept_presence"
    return "legacy_joint"


def _text_format_for_task_type(task_type: str):
    if task_type == "concept_presence":
        return ConceptPresenceDecision
    if task_type == "spatial_faithfulness":
        return SpatialFaithfulnessDecision
    return JudgeDecision


def _summarize(rows: list[dict[str, Any]], model: str, tasks_jsonl: Path) -> dict[str, Any]:
    task_types = Counter(row.get("task_type", _infer_task_type(row)) for row in rows)
    annotation_present = Counter(str(bool(row["metadata"]["annotation_present"])) for row in rows)
    output = {
        "metadata": {
            "model": model,
            "tasks_jsonl": str(tasks_jsonl),
            "num_tasks_judged": len(rows),
            "task_type_counts": dict(sorted(task_types.items())),
        },
        "annotation_present_counts": dict(sorted(annotation_present.items())),
    }
    if rows and all("concept_present" in row["judge"] for row in rows):
        presence = Counter(row["judge"]["concept_present"] for row in rows)
        mean_presence_conf = sum(float(row["judge"]["concept_present_confidence"]) for row in rows) / len(rows)
        output["concept_present_counts"] = dict(sorted(presence.items()))
        output["mean_concept_present_confidence"] = mean_presence_conf
    if rows and all("region_matches_concept" in row["judge"] for row in rows):
        region = Counter(row["judge"]["region_matches_concept"] for row in rows)
        mean_region_conf = sum(float(row["judge"]["region_matches_concept_confidence"]) for row in rows) / len(rows)
        output["region_match_counts"] = dict(sorted(region.items()))
        output["mean_region_matches_concept_confidence"] = mean_region_conf
    return output


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    output_jsonl, output_summary = _default_output_paths(args.tasks_jsonl)
    if args.output_jsonl is not None:
        output_jsonl = args.output_jsonl
    if args.output_summary is not None:
        output_summary = args.output_summary

    tasks = _load_jsonl(args.tasks_jsonl)
    if args.task_id_prefix:
        tasks = [task for task in tasks if str(task.get("task_id", "")).startswith(args.task_id_prefix)]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    existing = _load_existing(output_jsonl, args.overwrite)
    pending = [task for task in tasks if task["task_id"] not in existing]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Running VLM judge with model={args.model}. total_tasks={len(tasks)} pending={len(pending)}",
        flush=True,
    )

    client = OpenAI()
    base_dir = args.tasks_jsonl.parent

    with output_jsonl.open("a") as out_f:
        for idx, task in enumerate(pending, start=1):
            last_error: Exception | None = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    parsed, response = _call_openai(client, args.model, task, base_dir)
                    record = {
                        "task_id": task["task_id"],
                        "task_type": _infer_task_type(task),
                        "model": args.model,
                        "image_file": task["image_file"],
                        "overlay_file": task.get("overlay_file"),
                        "concept_name": task["concept_name"],
                        "metadata": task["metadata"],
                        "judge": parsed.model_dump(),
                        "openai_response_id": getattr(response, "id", None),
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    existing[task["task_id"]] = record
                    judge = record["judge"]
                    status_parts = [f"[{idx}/{len(pending)}] judged task_id={task['task_id']}"]
                    if "concept_present" in judge:
                        status_parts.append(f"present={judge['concept_present']}")
                    if "region_matches_concept" in judge:
                        status_parts.append(f"region={judge['region_matches_concept']}")
                    print(
                        " ".join(status_parts),
                        flush=True,
                    )
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
                    break
                except Exception as exc:
                    last_error = exc
                    print(
                        f"[{idx}/{len(pending)}] attempt {attempt}/{args.max_retries} failed "
                        f"for task_id={task['task_id']}: {exc!r}",
                        flush=True,
                    )
                    if attempt == args.max_retries:
                        raise
                    time.sleep(max(1.0, args.sleep_seconds))
            if last_error is not None and task["task_id"] not in existing:
                raise last_error

    ordered = [existing[task["task_id"]] for task in tasks if task["task_id"] in existing]
    summary = _summarize(ordered, args.model, args.tasks_jsonl)
    output_summary.write_text(json.dumps(summary, indent=2))
    print(
        f"Done. judged={len(ordered)} summary={output_summary}",
        flush=True,
    )


if __name__ == "__main__":
    main()
