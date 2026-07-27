from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "data" / "evaluation_taxonomy.json"
EVALUATION_STRATEGY = "evaluation_strategy"
QUESTION_TYPE = "question_type"


def load_taxonomy(path: str | Path = DEFAULT_TAXONOMY_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def taxonomy_hits_for_entry(entry: dict[str, Any], taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    text = _entry_text(entry).lower()
    hits = []
    hits.extend(_hits_for_dimension(taxonomy.get("tags", []), text, EVALUATION_STRATEGY))
    hits.extend(_hits_for_dimension(taxonomy.get("question_types", []), text, QUESTION_TYPE))
    return hits


def _hits_for_dimension(
    tags: list[dict[str, Any]],
    text: str,
    dimension: str,
) -> list[dict[str, Any]]:
    hits = []
    for tag in tags:
        indicators = [indicator.lower() for indicator in tag.get("indicators", [])]
        negative_indicators = [
            indicator.lower() for indicator in tag.get("negative_indicators", [])
        ]
        if any(
            _indicator_matches(indicator, text)
            for indicator in negative_indicators
        ):
            continue
        matched = [indicator for indicator in indicators if _indicator_matches(indicator, text)]
        if not matched:
            continue
        hits.append(
            {
                "dimension": dimension,
                "tag": tag["id"],
                "label": tag.get("label", tag["id"]),
                "matched_indicators": matched[:5],
            }
        )
    return hits


def _indicator_matches(indicator: str, text: str) -> bool:
    if not indicator:
        return False
    if _is_word_or_phrase(indicator):
        pattern = rf"(?<![a-z0-9_]){re.escape(indicator)}(?![a-z0-9_])"
        return any(
            not _match_is_negated(text, match.start())
            for match in re.finditer(pattern, text)
        )
    return any(
        not _match_is_negated(text, index)
        for index in _substring_positions(indicator, text)
    )


def _match_is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 80) : start]
    return re.search(
        r"(?:\bdo\s+not|\bdon't|\bmust\s+not|\bno|\bwithout|"
        r"\bavoid(?:ing)?|\bforbid(?:den|s)?)"
        r"(?:\s+[\w-]+){0,3}\s*$",
        prefix,
    ) is not None


def _substring_positions(needle: str, text: str) -> list[int]:
    positions = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return positions
        positions.append(index)
        start = index + len(needle)


def _is_word_or_phrase(indicator: str) -> bool:
    return all(char.isalnum() or char.isspace() or char in {"-", "_"} for char in indicator)


def _entry_text(entry: dict[str, Any]) -> str:
    parsed = entry.get("parsed")
    metadata = entry.get("metadata") or {}
    structured_role = metadata.get("interaction_role") in {
        "assessment",
        "evidence_card",
        "round_ranking",
        "judge_ranking",
        "probe_comparison",
        "wave_judgment",
        "memory_update",
    }
    parts = [] if isinstance(parsed, dict) and structured_role else [str(entry.get("content", ""))]
    if isinstance(parsed, dict):
        for value in parsed.values():
            parts.append(_flatten_text(value))
    return "\n".join(part for part in parts if part)


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_flatten_text(item) for item in value.values())
    return str(value)
