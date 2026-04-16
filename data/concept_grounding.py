"""Concept localizability and prompt normalization helpers for mask generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from data.sam3_concept_mask_cache import canonicalize_concept_label


LOCALIZABILITY_LOCAL = "part_localizable"
LOCALIZABILITY_REGION = "region_localizable"
LOCALIZABILITY_MULTI = "multi_region_ambiguous"
LOCALIZABILITY_NONVISUAL = "non_localizable"


_DEFAULT_OVERRIDES: Dict[str, Dict[str, str]] = {
    "long black bill": {
        "normalized_prompt": "bird bill",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "yellow bill": {
        "normalized_prompt": "bird bill",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "small beak": {
        "normalized_prompt": "bird beak",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a long curved bill": {
        "normalized_prompt": "bird bill",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a black wingtip": {
        "normalized_prompt": "bird wingtip",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a white patch on the wing": {
        "normalized_prompt": "bird wing patch",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "white wingbars": {
        "normalized_prompt": "bird wingbars",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a white wingbar": {
        "normalized_prompt": "bird wingbar",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "wing bars": {
        "normalized_prompt": "bird wing bars",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a long tail with white bars": {
        "normalized_prompt": "bird tail",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a yellow tipped tail": {
        "normalized_prompt": "bird tail",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "shiny black eyes": {
        "normalized_prompt": "bird eye",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a red eye": {
        "normalized_prompt": "bird eye",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a red ring around the eye": {
        "normalized_prompt": "bird eye ring",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a white eye ring": {
        "normalized_prompt": "bird eye ring",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a white stripe above the eye": {
        "normalized_prompt": "bird eyebrow stripe",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a black mask around the eyes": {
        "normalized_prompt": "bird eye mask",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a black mask over its eyes": {
        "normalized_prompt": "bird eye mask",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a black mask over the eyes": {
        "normalized_prompt": "bird eye mask",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a black face": {
        "normalized_prompt": "bird face",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a black cheek patch": {
        "normalized_prompt": "bird cheek patch",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a red crest on the head": {
        "normalized_prompt": "bird crest",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a crest on the head": {
        "normalized_prompt": "bird crest",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a red head": {
        "normalized_prompt": "bird head",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "orange legs": {
        "normalized_prompt": "bird legs",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "red legs": {
        "normalized_prompt": "bird legs",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "long black legs": {
        "normalized_prompt": "bird legs",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "black legs and feet": {
        "normalized_prompt": "bird legs and feet",
        "localizability": LOCALIZABILITY_LOCAL,
    },
    "a wingspan of up to 3 5 feet": {
        "normalized_prompt": "bird",
        "localizability": LOCALIZABILITY_NONVISUAL,
    },
    "a wingspan of 3 4 feet": {
        "normalized_prompt": "bird",
        "localizability": LOCALIZABILITY_NONVISUAL,
    },
    "a raucous call": {
        "normalized_prompt": "bird",
        "localizability": LOCALIZABILITY_NONVISUAL,
    },
    "fast erratic flight patterns": {
        "normalized_prompt": "bird",
        "localizability": LOCALIZABILITY_NONVISUAL,
    },
    "pattern": {
        "normalized_prompt": "bird plumage",
        "localizability": LOCALIZABILITY_MULTI,
    },
    "color": {
        "normalized_prompt": "bird plumage",
        "localizability": LOCALIZABILITY_MULTI,
    },
}


@dataclass(frozen=True)
class ConceptGroundingSpec:
    concept_index: int
    raw_concept: str
    normalized_concept: str
    normalized_prompt: str
    localizability: str
    keep_for_masking: bool


def _guess_localizability(concept: str) -> str:
    nonvisual_tokens = {
        "call",
        "voice",
        "behavior",
        "flight",
        "wingspan",
        "hardness",
        "dispersion",
        "keyboard",
        "spider",
        "antennae",
        "toes",
    }
    local_tokens = {
        "bill",
        "beak",
        "eye",
        "eyes",
        "eyebrow",
        "throat",
        "breast",
        "belly",
        "head",
        "face",
        "cheek",
        "crest",
        "wing",
        "wingbar",
        "wingbars",
        "wingtip",
        "tail",
        "neck",
        "legs",
        "feet",
        "foot",
        "back",
        "cap",
        "ring",
        "mask",
        "patch",
        "stripe",
        "body",
    }
    multi_tokens = {"underparts", "upperparts", "plumage", "feathers", "coloration", "color", "pattern"}
    tokens = set(concept.split())
    if tokens & nonvisual_tokens:
        return LOCALIZABILITY_NONVISUAL
    if tokens & multi_tokens:
        return LOCALIZABILITY_MULTI
    if tokens & local_tokens:
        if "body" in tokens or "plumage" in tokens:
            return LOCALIZABILITY_REGION
        return LOCALIZABILITY_LOCAL
    return LOCALIZABILITY_MULTI


def _default_prompt(concept: str, localizability: str) -> str:
    if localizability == LOCALIZABILITY_LOCAL:
        return f"bird {concept}"
    if localizability == LOCALIZABILITY_REGION:
        return f"bird {concept}"
    if localizability == LOCALIZABILITY_MULTI:
        return f"bird region for {concept}"
    return "bird"


def resolve_grounding_spec(
    concept: str,
    concept_index: int,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    allowed_localizability: Optional[Sequence[str]] = None,
) -> ConceptGroundingSpec:
    normalized = canonicalize_concept_label(concept)
    merged = dict(_DEFAULT_OVERRIDES)
    if overrides:
        for key, value in overrides.items():
            merged[canonicalize_concept_label(key)] = dict(value)
    override = merged.get(normalized, {})
    localizability = str(override.get("localizability") or _guess_localizability(normalized))
    normalized_prompt = str(override.get("normalized_prompt") or _default_prompt(normalized, localizability))
    allowed = set(allowed_localizability or [LOCALIZABILITY_LOCAL, LOCALIZABILITY_REGION])
    keep = localizability in allowed
    return ConceptGroundingSpec(
        concept_index=int(concept_index),
        raw_concept=concept,
        normalized_concept=normalized,
        normalized_prompt=normalized_prompt,
        localizability=localizability,
        keep_for_masking=keep,
    )


def build_grounding_specs(
    concepts: Sequence[str],
    concept_indices: Iterable[int],
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    allowed_localizability: Optional[Sequence[str]] = None,
) -> List[ConceptGroundingSpec]:
    return [
        resolve_grounding_spec(
            concept=concepts[int(concept_index)],
            concept_index=int(concept_index),
            overrides=overrides,
            allowed_localizability=allowed_localizability,
        )
        for concept_index in concept_indices
    ]
