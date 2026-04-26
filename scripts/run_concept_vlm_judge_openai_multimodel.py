#!/usr/bin/env python3
"""Run an OpenAI concept-presence judge once per image across multiple CBM task files."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field


SYSTEM_PROMPT = """You are a careful multimodal judge for a concept-bottleneck interpretability study.

You will receive:
- one original image
- a list of candidate concepts produced by multiple models

Your job is to judge whether each concept is visibly present anywhere in the image.
Be conservative and use "unsure" when the evidence is weak.
"""


class BatchedConceptDecision(BaseModel):
    concept_name: str
    concept_present: str = Field(pattern="^(yes|no|unsure)$")
    concept_present_confidence: float = Field(ge=0.0, le=1.0)
    rationale_short: str


class BatchedConceptResponse(BaseModel):
    judgments: list[BatchedConceptDecision]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-jsonl",
        action="append",
        dest="tasks_jsonls",
        required=True,
        help="Input task JSONL. Pass once per model export.",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _to_jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return str(value)


def _extract_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    usage_json = _to_jsonable(usage)
    return usage_json if isinstance(usage_json, dict) else {"raw": usage_json}


def _accumulate_usage(totals: dict[str, Any], usage: dict[str, Any]) -> None:
    for key, value in usage.items():
        if isinstance(value, dict):
            child = totals.setdefault(key, {})
            if isinstance(child, dict):
                _accumulate_usage(child, value)
            else:
                totals[key] = _to_jsonable(value)
        elif isinstance(value, bool):
            totals[key] = int(totals.get(key, 0)) + int(value)
        elif isinstance(value, (int, float)):
            totals[key] = totals.get(key, 0) + value
        elif key not in totals:
            totals[key] = value


def _resolve_path(base_dir: Path, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _image_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        mime = "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{payload}"


def _normalize_concept_name(name: str) -> str:
    return " ".join(name.lower().strip().split())


def _group_tasks(tasks_jsonls: list[Path]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for tasks_jsonl in tasks_jsonls:
        base_dir = tasks_jsonl.parent
        for row in _load_jsonl(tasks_jsonl):
            if row.get("task_type") != "concept_presence":
                continue
            dataset_index = int(row["dataset_index"])
            item = grouped.setdefault(
                dataset_index,
                {
                    "dataset_index": dataset_index,
                    "image_file": row["image_file"],
                    "base_dir": base_dir,
                    "concepts": defaultdict(list),
                },
            )
            key = _normalize_concept_name(row["concept_name"])
            item["concepts"][key].append(row)
    ordered = [grouped[idx] for idx in sorted(grouped)]
    return ordered


def _build_prompt(image_group: dict[str, Any]) -> str:
    lines = [
        "You are evaluating concept presence for one image.",
        "",
        "Multiple concept bottleneck models proposed the candidate concepts below.",
        "Judge each unique concept once based only on visible evidence in the image.",
        'Use "unsure" when the concept is subtle, ambiguous, occluded, or resolution is insufficient.',
        "",
        "Return one judgment per concept_name.",
        "",
        "Candidate concepts:",
    ]
    concept_names = []
    for sources in image_group["concepts"].values():
        concept_name = sources[0]["concept_name"]
        model_names = sorted({str(row.get("model_name", "unknown")) for row in sources})
        lines.append(f"- {concept_name} (proposed by: {', '.join(model_names)})")
        concept_names.append(concept_name)
    return "\n".join(lines), concept_names


def _call_openai(client: OpenAI, model: str, image_group: dict[str, Any]) -> tuple[BatchedConceptResponse, Any]:
    prompt, _ = _build_prompt(image_group)
    image_path = _resolve_path(image_group["base_dir"], image_group["image_file"])
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_text", "text": "Original image:"},
                    {"type": "input_image", "image_url": _image_to_data_url(image_path)},
                ],
            },
        ],
        text_format=BatchedConceptResponse,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError(f"No parsed output for dataset_index={image_group['dataset_index']}")
    return parsed, response


def _summarize(rows: list[dict[str, Any]], model: str, tasks_jsonls: list[Path]) -> dict[str, Any]:
    response_ids = {row.get("openai_response_id") for row in rows if row.get("openai_response_id")}
    usage_totals: dict[str, Any] = {}
    seen_usage_ids: set[str] = set()
    for idx, row in enumerate(rows):
        usage = row.get("openai_usage")
        if not isinstance(usage, dict):
            continue
        response_id = row.get("openai_response_id")
        dedupe_key = str(response_id) if response_id else f"row:{idx}"
        if dedupe_key in seen_usage_ids:
            continue
        seen_usage_ids.add(dedupe_key)
        _accumulate_usage(usage_totals, usage)

    by_model = defaultdict(Counter)
    for row in rows:
        by_model[str(row["source_model"])][str(row["judge"]["concept_present"])] += 1

    summary = {
        "metadata": {
            "model": model,
            "tasks_jsonls": [str(p) for p in tasks_jsonls],
            "num_rows": len(rows),
            "num_unique_openai_responses": len(response_ids),
        },
        "per_model_concept_present_counts": {
            model_name: dict(sorted(counter.items()))
            for model_name, counter in sorted(by_model.items())
        },
    }
    if usage_totals:
        summary["usage_totals"] = usage_totals
    return summary


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    if args.output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_jsonl} already exists; pass --overwrite to replace it.")

    groups = _group_tasks([Path(p) for p in args.tasks_jsonls])
    if args.max_images is not None:
        groups = groups[: args.max_images]

    print(
        f"Running multimodel VLM judge with model={args.model}. images={len(groups)} task_files={len(args.tasks_jsonls)}",
        flush=True,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    all_rows: list[dict[str, Any]] = []
    with args.output_jsonl.open("w") as out_f:
        for idx, image_group in enumerate(groups, start=1):
            last_error: Exception | None = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    parsed, response = _call_openai(client, args.model, image_group)
                    judgments = {
                        _normalize_concept_name(item.concept_name): item
                        for item in parsed.judgments
                    }
                    expected_keys = set(image_group["concepts"])
                    if set(judgments) != expected_keys:
                        missing = sorted(expected_keys - set(judgments))
                        extra = sorted(set(judgments) - expected_keys)
                        raise RuntimeError(
                            f"Concept mismatch for dataset_index={image_group['dataset_index']}: "
                            f"missing={missing} extra={extra}"
                        )
                    usage = _extract_usage(response)
                    for norm_name, source_rows in image_group["concepts"].items():
                        item = judgments[norm_name]
                        for source in source_rows:
                            record = {
                                "task_id": source["task_id"],
                                "dataset_index": source["dataset_index"],
                                "image_file": source["image_file"],
                                "source_model": source["model_name"],
                                "concept_name": source["concept_name"],
                                "judge": item.model_dump(),
                                "openai_response_id": getattr(response, "id", None),
                                "openai_usage": usage,
                            }
                            out_f.write(json.dumps(record) + "\n")
                            all_rows.append(record)
                    out_f.flush()
                    print(
                        f"[{idx}/{len(groups)}] judged dataset_index={image_group['dataset_index']} "
                        f"unique_concepts={len(image_group['concepts'])}",
                        flush=True,
                    )
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
                    break
                except Exception as exc:
                    last_error = exc
                    print(
                        f"[{idx}/{len(groups)}] attempt {attempt}/{args.max_retries} failed "
                        f"for dataset_index={image_group['dataset_index']}: {exc!r}",
                        flush=True,
                    )
                    if attempt == args.max_retries:
                        raise
                    time.sleep(max(1.0, args.sleep_seconds))
            if last_error is not None and not any(row["dataset_index"] == image_group["dataset_index"] for row in all_rows):
                raise last_error

    summary = _summarize(all_rows, args.model, [Path(p) for p in args.tasks_jsonls])
    args.output_summary.write_text(json.dumps(summary, indent=2))
    print(f"Done. rows={len(all_rows)} summary={args.output_summary}", flush=True)


if __name__ == "__main__":
    main()
