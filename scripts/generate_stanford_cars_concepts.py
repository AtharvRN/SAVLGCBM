#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stanford_cars_common import canonicalize_concept_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stanford Cars concept vocabularies.")
    parser.add_argument("--raw_output", default="concept_files/stanford_cars_concepts_raw.txt")
    parser.add_argument("--filtered_output", default="concept_files/stanford_cars_concepts_filtered.txt")
    parser.add_argument("--metadata_output", default="data/stanford_cars/concept_metadata.json")
    return parser.parse_args()


def dedupe(items: Iterable[Tuple[str, str, bool]]) -> List[Dict[str, object]]:
    seen: set[str] = set()
    out: List[Dict[str, object]] = []
    for label, category, keep in items:
        canonical = canonicalize_concept_label(label)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        out.append(
            {
                "label": label,
                "canonical_label": canonical,
                "prompt": label,
                "category": category,
                "keep": bool(keep),
            }
        )
    return out


def add_many(labels: Sequence[str], category: str, keep: bool = True) -> List[Tuple[str, str, bool]]:
    return [(label, category, keep) for label in labels]


def expand(position_tokens: Sequence[str], parts: Sequence[str], category: str, keep: bool = True) -> List[Tuple[str, str, bool]]:
    out: List[Tuple[str, str, bool]] = []
    for token in position_tokens:
        for part in parts:
            label = f"{token} {part}".strip()
            out.append((label, category, keep))
    return out


def build_concepts() -> List[Dict[str, object]]:
    front_parts = [
        "front grille",
        "lower grille",
        "upper grille",
        "mesh grille",
        "chrome grille",
        "black grille",
        "front bumper",
        "front splitter",
        "air intake",
        "hood",
        "hood scoop",
        "windshield",
        "windshield wiper",
        "front license plate",
        "grille emblem",
        "fog light",
        "daytime running light",
        "front badge",
    ]
    side_parts = [
        "wheel",
        "alloy wheel",
        "rim",
        "tire",
        "wheel arch",
        "side mirror",
        "door handle",
        "door seam",
        "side window",
        "quarter window",
        "roofline",
        "roof rail",
        "roof rack",
        "sunroof",
        "pillar",
        "side skirt",
        "rocker panel",
        "fender vent",
    ]
    rear_parts = [
        "taillight",
        "rear bumper",
        "rear diffuser",
        "trunk lid",
        "rear window",
        "license plate recess",
        "exhaust pipe",
        "dual exhaust",
        "spoiler",
        "roof spoiler",
        "rear badge",
        "rear hatch",
    ]
    body_styles = [
        "sedan body",
        "coupe body",
        "convertible roof",
        "soft top",
        "hardtop convertible",
        "pickup bed",
        "hatchback rear",
        "SUV body",
        "minivan body",
        "wagon body",
        "fastback roofline",
        "notchback rear",
        "liftback rear",
        "boxy SUV body",
        "sloping roofline",
        "long roof wagon",
        "short rear deck",
        "long hood",
    ]
    visual_attributes = [
        "two doors",
        "four doors",
        "round headlights",
        "rectangular headlights",
        "thin headlights",
        "horizontal headlights",
        "vertical taillights",
        "horizontal taillights",
        "thin taillights",
        "wide taillights",
        "wide grille",
        "narrow grille",
        "large grille opening",
        "chrome window trim",
        "black window trim",
        "roof rack",
        "panoramic roof",
        "sunroof panel",
        "arched roofline",
        "upright windshield",
        "low hood",
        "raised hood",
        "flared fenders",
        "pronounced wheel arches",
        "large wheels",
        "small wheels",
        "black wheels",
        "silver wheels",
        "chrome wheels",
        "short wheelbase look",
        "long wheelbase look",
        "squared rear end",
        "rounded rear end",
        "integrated spoiler",
        "flush door handles",
    ]
    brand_cues = [
        "BMW kidney grille",
        "Mercedes grille emblem",
        "Mercedes hood ornament",
        "Jeep seven slot grille",
        "Porsche round headlights",
        "Mini rounded body",
        "Audi single frame grille",
        "Lexus spindle grille",
        "Subaru hood scoop",
        "Ford Mustang grille",
        "Dodge crosshair grille",
        "Chevrolet bowtie badge",
        "Bentley mesh grille",
        "Rolls Royce grille",
        "Land Rover badge",
        "Maserati grille trident",
    ]
    color_concepts = [
        "red car paint",
        "blue car paint",
        "black car paint",
        "white car paint",
        "silver car paint",
        "gray car paint",
    ]

    items: List[Tuple[str, str, bool]] = []
    items.extend(add_many(front_parts, "front_parts", keep=True))
    items.extend(add_many(side_parts, "side_parts", keep=True))
    items.extend(add_many(rear_parts, "rear_parts", keep=True))
    items.extend(add_many(body_styles, "body_style", keep=True))
    items.extend(add_many(visual_attributes, "visual_attributes", keep=True))
    items.extend(add_many(brand_cues, "brand_cues", keep=True))
    items.extend(add_many(color_concepts, "color", keep=False))

    items.extend(expand(["left", "right"], ["headlight", "fog light", "side mirror", "taillight"], "laterality", keep=True))
    items.extend(expand(["left", "right"], ["door handle", "side window", "quarter window", "exhaust pipe"], "laterality", keep=True))
    items.extend(expand(["front left", "front right", "rear left", "rear right"], ["wheel", "rim", "tire"], "wheel_position", keep=True))
    items.extend(expand(["front left", "front right"], ["headlight", "fog light", "fender"], "front_corner", keep=True))
    items.extend(expand(["rear left", "rear right"], ["taillight", "exhaust pipe", "quarter panel"], "rear_corner", keep=True))
    items.extend(expand(["front", "rear"], ["wheel", "bumper", "fender"], "body_region", keep=True))
    items.extend(expand(["front", "rear"], ["license plate", "window", "badge"], "body_region", keep=True))
    items.extend(expand(["upper", "lower"], ["bumper", "window line", "spoiler"], "body_subregion", keep=False))
    items.extend(expand(["upper", "lower"], ["grille", "air intake"], "front_subpart", keep=True))
    items.extend(expand(["chrome", "black", "mesh"], ["front grille", "window trim"], "appearance_modifier", keep=True))
    items.extend(expand(["chrome", "black", "painted"], ["side mirror", "door handle", "wheel"], "appearance_modifier", keep=False))
    items.extend(expand(["wide", "narrow", "tall", "low"], ["grille", "headlights"], "shape_modifier", keep=False))
    items.extend(expand(["wide", "narrow", "thin"], ["taillights", "air intake", "roofline"], "shape_modifier", keep=False))
    items.extend(expand(["rounded", "boxy", "squared", "sloping"], ["roofline", "rear end"], "shape_descriptor", keep=False))
    items.extend(expand(["arched", "flat", "tapered"], ["roofline", "hood", "rear window"], "shape_descriptor", keep=False))
    items.extend(expand(["single", "dual"], ["exhaust pipe"], "rear_exhaust", keep=True))
    items.extend(expand(["small", "large"], ["rear spoiler", "sunroof", "grille opening"], "scale_descriptor", keep=False))
    items.extend(expand(["small", "large"], ["wheels", "wheel arches", "headlights", "taillights"], "scale_descriptor", keep=False))
    items.extend(expand(["prominent", "subtle"], ["hood scoop", "wheel arch", "rear spoiler"], "style_descriptor", keep=False))

    exploratory = [
        "front fascia",
        "rear fascia",
        "beltline",
        "greenhouse",
        "c pillar",
        "d pillar",
        "quarter panel",
        "rear quarter window",
        "front overhang",
        "rear overhang",
        "front lip",
        "side cladding",
        "body cladding",
        "black side cladding",
        "chrome door trim",
        "body color mirror cap",
        "black mirror cap",
        "wraparound taillight",
        "projector headlight",
        "LED light strip",
        "rear light bar",
        "split grille",
        "large lower intake",
        "small upper grille",
        "curved hood",
        "flat hood",
        "bulging fender",
        "muscular rear haunch",
        "upright body",
        "low roof",
        "tapered roofline",
        "rear hatch glass",
        "integrated trunk lip",
        "roof antenna",
        "roof rails",
        "roof cross bars",
        "spare tire cover",
        "pickup tailgate",
        "cargo area glass",
        "rear quarter flare",
        "black bumper insert",
        "chrome bumper trim",
        "silver skid plate",
        "running board",
        "side step",
        "black wheel arch trim",
        "painted fender flare",
        "window visor",
        "license plate frame",
        "hood ornament",
        "round fog lights",
        "rectangular fog lights",
        "multi spoke wheel",
        "five spoke wheel",
        "split spoke wheel",
        "black alloy wheel",
        "silver alloy wheel",
        "chrome alloy wheel",
        "mesh wheel",
        "large brake opening",
        "front tow hook cover",
        "rear reflector",
        "rear valance",
        "rear wiper",
        "trapezoid grille",
        "hexagonal grille",
        "oval grille",
        "boxy silhouette",
        "sleek silhouette",
        "upright front end",
        "sloped front end",
        "stubby rear overhang",
        "extended rear overhang",
        "short hood",
        "long rear overhang",
        "short rear overhang",
        "upright tailgate",
        "sloped tailgate",
        "thick c pillar",
        "thin c pillar",
        "large quarter window",
        "small quarter window",
        "blacked out pillar",
        "floating roof look",
        "body colored door handles",
        "chrome door handles",
        "body colored bumpers",
        "black bumpers",
        "silver roof rails",
        "high ground clearance",
        "low ground clearance",
        "raised ride height",
        "sporty stance",
        "luxury styling",
        "expensive car",
        "fast car",
    ]
    items.extend(add_many(exploratory, "exploratory", keep=False))
    return dedupe(items)


def main() -> None:
    args = parse_args()
    raw_output = Path(args.raw_output).resolve()
    filtered_output = Path(args.filtered_output).resolve()
    metadata_output = Path(args.metadata_output).resolve()
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    filtered_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    concepts = build_concepts()
    raw_lines = [str(item["label"]) for item in concepts]
    filtered_lines = [str(item["label"]) for item in concepts if bool(item["keep"])]

    raw_output.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    filtered_output.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")
    metadata = {
        "raw_count": len(raw_lines),
        "filtered_count": len(filtered_lines),
        "raw_output": str(raw_output),
        "filtered_output": str(filtered_output),
        "concepts": concepts,
    }
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
