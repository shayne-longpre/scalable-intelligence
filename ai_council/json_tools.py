from __future__ import annotations

import json
import re
from typing import Any


class JsonExtractionError(ValueError):
    """Raised when a model response does not contain one parseable JSON object."""


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_fenced_json(stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        candidate = _find_balanced_object(stripped)
        if candidate is None:
            raise JsonExtractionError("no JSON object found in response")
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise JsonExtractionError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise JsonExtractionError("expected a JSON object")
    return parsed


def _strip_fenced_json(text: str) -> str:
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else text


def _find_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
