#!/usr/bin/env python3
"""Map CUB concepts to part annotations using GPT-5.4.

This script asks an LLM to decide whether a concept can be grounded to one of
the CUB part annotations and, if so, which part group it belongs to. It writes:

1. a full JSON artifact with per-concept decisions
2. a filtered TXT file containing only part-aligned concepts
3. a CSV summary for quick inspection

Typical usage:

    python scripts/map_cub_concepts_to_parts_openai.py \
      --concept-file concept_files/cub_filtered.txt \
      --parts-file ~/Downloads/CUB_200_2011/parts/parts.txt \
      --output-json results/cub_concept_part_mapping_gpt54.json \
      --output-filtered results/cub_part_aligned_concepts_gpt54.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from pydantic import BaseModel, Field


COARSE_TO_EXACT_PARTS: dict[str, list[str]] = {
    "back": ["back"],
    "beak": ["beak"],
    "belly": ["belly"],
    "breast": ["breast"],
    "crown": ["crown"],
    "forehead": ["forehead"],
    "eye": ["left eye", "right eye"],
    "leg": ["left leg", "right leg"],
    "wing": ["left wing", "right wing"],
    "nape": ["nape"],
    "tail": ["tail"],
    "throat": ["throat"],
}


class ConceptDecision(BaseModel):
    concept: str = Field(description="The exact concept string from the input list.")
    keep: bool = Field(
        description=(
            "True only if this concept can be reliably aligned to a CUB part "
            "annotation for localization."
        )
    )
    part_group: Optional[str] = Field(
        description=(
            "One coarse part group from the allowed set, or null if the concept "
            "does not align cleanly to a supported part."
        )
    )
    rationale: str = Field(
        description=(
            "A short explanation. Focus on whether the concept corresponds to a "
            "supported part annotation."
        )
    )


class BatchDecision(BaseModel):
    mappings: list[ConceptDecision]


SYSTEM_PROMPT = """You are mapping bird concepts to the CUB part annotation system.

Your job is binary:
1. keep concepts that can be reliably localized using the available CUB parts
2. reject concepts that do not align cleanly to those parts

Available coarse part groups:
- back
- beak
- belly
- breast
- crown
- forehead
- eye
- leg
- wing
- nape
- tail
- throat

Rules:
- Only keep a concept if it primarily describes one supported part group.
- If the concept refers to a symmetric bilateral part, use the coarse group:
  - eye -> left eye + right eye
  - leg -> left leg + right leg
  - wing -> left wing + right wing
- Treat "bill" as beak.
- Reject concepts that describe:
  - whole bird / overall size
  - multiple parts at once
  - vague regions like "upperparts", "underparts", "body", "face", "shoulders"
  - unsupported parts such as feet if they cannot be mapped cleanly
  - abstract or non-spatial traits like shape, behavior, or global color pattern
- Be conservative. If the match is weak, reject it.

Examples:
- "yellow bill" -> keep, beak
- "black beak" -> keep, beak
- "yellow eyes" -> keep, eye
- "dark wingtips" -> reject (too specific relative to whole wing annotation)
- "all black body" -> reject
- "white underparts" -> reject
- "red shoulders" -> reject
- "long black tail" -> keep, tail
- "long legs" -> keep, leg
- "webbed feet" -> reject

Return only structured data matching the schema.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--concept-file",
        type=Path,
        default=Path("concept_files/cub_filtered.txt"),
        help="Concept list to map. Defaults to the repo's CUB filtered concept bank.",
    )
    parser.add_argument(
        "--parts-file",
        type=Path,
        default=Path.home() / "Downloads" / "CUB_200_2011" / "parts" / "parts.txt",
        help="CUB parts.txt file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/cub_concept_part_mapping_gpt54.json"),
        help="Full JSON artifact path.",
    )
    parser.add_argument(
        "--output-filtered",
        type=Path,
        default=Path("results/cub_part_aligned_concepts_gpt54.txt"),
        help="Filtered TXT of kept concepts.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/cub_concept_part_mapping_gpt54.csv"),
        help="CSV summary path.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4",
        help="OpenAI model name. Defaults to gpt-5.4.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of concepts per API call.",
    )
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=None,
        help="Optional limit for debugging or smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing partial output and start over.",
    )
    return parser.parse_args()


def load_cub_parts(parts_file: Path) -> dict[str, list[str]]:
    if not parts_file.exists():
        raise FileNotFoundError(f"parts file not found: {parts_file}")

    exact_parts: list[str] = []
    for line in parts_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        _, name = line.split(" ", 1)
        exact_parts.append(name.strip())

    expected = {part for parts in COARSE_TO_EXACT_PARTS.values() for part in parts}
    found = set(exact_parts)
    missing = sorted(expected - found)
    if missing:
        raise ValueError(
            "parts.txt is missing expected CUB parts: " + ", ".join(missing)
        )
    return COARSE_TO_EXACT_PARTS


def load_concepts(concept_file: Path, max_concepts: Optional[int]) -> list[str]:
    seen: set[str] = set()
    concepts: list[str] = []
    for line in concept_file.read_text().splitlines():
        concept = line.strip()
        if not concept or concept in seen:
            continue
        seen.add(concept)
        concepts.append(concept)
    if max_concepts is not None:
        concepts = concepts[:max_concepts]
    return concepts


def build_user_prompt(concepts: list[str], parts_map: dict[str, list[str]]) -> str:
    coarse_parts = ", ".join(sorted(parts_map))
    concept_lines = "\n".join(f"- {concept}" for concept in concepts)
    return f"""Map the following CUB bird concepts to the supported coarse part groups.

Supported coarse part groups:
{coarse_parts}

Concepts:
{concept_lines}

For each concept:
- set keep=true only if it aligns cleanly to one supported part group
- otherwise set keep=false and part_group=null
- use the exact concept string
- keep the rationale short and concrete
"""


def call_batch(
    client: OpenAI,
    model: str,
    concepts: list[str],
    parts_map: dict[str, list[str]],
) -> list[dict[str, Any]]:
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(concepts, parts_map)},
        ],
        text_format=BatchDecision,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("OpenAI response did not produce parsed output.")

    returned = [item.concept for item in parsed.mappings]
    missing = sorted(set(concepts) - set(returned))
    extras = sorted(set(returned) - set(concepts))
    if missing or extras:
        raise ValueError(
            "Model response concept mismatch. "
            f"missing={missing[:10]} extras={extras[:10]}"
        )

    concept_to_decision = {item.concept: item for item in parsed.mappings}
    normalized: list[dict[str, Any]] = []
    for concept in concepts:
        item = concept_to_decision[concept]
        part_group = item.part_group
        if part_group is not None and part_group not in parts_map:
            raise ValueError(
                f"Invalid part_group {part_group!r} for concept {concept!r}"
            )
        keep = bool(item.keep and part_group is not None)
        normalized.append(
            {
                "concept": concept,
                "keep": keep,
                "part_group": part_group if keep else None,
                "exact_parts": parts_map[part_group] if keep else [],
                "rationale": item.rationale.strip(),
            }
        )
    return normalized


def load_existing(output_json: Path, overwrite: bool) -> dict[str, dict[str, Any]]:
    if overwrite or not output_json.exists():
        return {}
    payload = json.loads(output_json.read_text())
    mappings = payload.get("mappings", [])
    return {item["concept"]: item for item in mappings}


def save_outputs(
    output_json: Path,
    output_filtered: Path,
    output_csv: Path,
    concept_file: Path,
    parts_file: Path,
    model: str,
    batch_size: int,
    mappings: list[dict[str, Any]],
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_filtered.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    kept = [item["concept"] for item in mappings if item["keep"]]
    counts = Counter(item["part_group"] for item in mappings if item["keep"])
    payload = {
        "meta": {
            "model": model,
            "concept_file": str(concept_file),
            "parts_file": str(parts_file),
            "batch_size": batch_size,
            "num_concepts": len(mappings),
            "num_kept": len(kept),
            "part_group_counts": dict(sorted(counts.items())),
        },
        "mappings": mappings,
    }
    output_json.write_text(json.dumps(payload, indent=2))
    output_filtered.write_text("\n".join(kept) + ("\n" if kept else ""))

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["concept", "keep", "part_group", "exact_parts", "rationale"],
        )
        writer.writeheader()
        for item in mappings:
            writer.writerow(
                {
                    "concept": item["concept"],
                    "keep": item["keep"],
                    "part_group": item["part_group"] or "",
                    "exact_parts": "|".join(item["exact_parts"]),
                    "rationale": item["rationale"],
                }
            )


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    parts_map = load_cub_parts(args.parts_file)
    concepts = load_concepts(args.concept_file, args.max_concepts)
    existing = load_existing(args.output_json, args.overwrite)

    client = OpenAI()
    concept_to_mapping = dict(existing)
    pending = [concept for concept in concepts if concept not in concept_to_mapping]

    if pending:
        print(
            f"Mapping {len(concepts)} concepts with model={args.model}, "
            f"batch_size={args.batch_size}. Pending={len(pending)}",
            flush=True,
        )
    else:
        print("All concepts already mapped; rewriting outputs.", flush=True)

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        print(
            f"Requesting batch {start // args.batch_size + 1}: "
            f"{len(batch)} concepts",
            flush=True,
        )
        batch_results = call_batch(client, args.model, batch, parts_map)
        for item in batch_results:
            concept_to_mapping[item["concept"]] = item

        ordered = [concept_to_mapping[concept] for concept in concepts if concept in concept_to_mapping]
        save_outputs(
            args.output_json,
            args.output_filtered,
            args.output_csv,
            args.concept_file,
            args.parts_file,
            args.model,
            args.batch_size,
            ordered,
        )
        kept = sum(1 for item in ordered if item["keep"])
        print(
            f"Saved partial outputs: mapped={len(ordered)} kept={kept}",
            flush=True,
        )

    final_mappings = [concept_to_mapping[concept] for concept in concepts]
    save_outputs(
        args.output_json,
        args.output_filtered,
        args.output_csv,
        args.concept_file,
        args.parts_file,
        args.model,
        args.batch_size,
        final_mappings,
    )
    kept = sum(1 for item in final_mappings if item["keep"])
    print(
        f"Done. total={len(final_mappings)} kept={kept} "
        f"output={args.output_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
