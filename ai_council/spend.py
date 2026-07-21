from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Any


def compute_spend_lineage(run_dir: str | Path) -> dict[str, Any]:
    """Aggregate incremental spend across a run and its replay sources."""

    root = Path(run_dir).resolve()
    pending = deque([root])
    visited: set[Path] = set()
    unresolved: set[str] = set()
    transcript_fallbacks: set[str] = set()
    model_spend: dict[str, dict[str, Any]] = {}
    model_calls = 0
    reported_cost = 0.0

    while pending:
        current = pending.popleft().resolve()
        if current in visited:
            continue
        visited.add(current)
        summary = _load_object(current / "run_summary.json")
        if not summary:
            fallback = _spend_from_transcript(current / "transcript.jsonl")
            model_calls += fallback["model_calls"]
            reported_cost += fallback["reported_cost_usd"]
            _merge_model_spend(model_spend, fallback["model_spend"])
            transcript_fallbacks.add(str(current))
        else:
            model_calls += _nonnegative_int(summary.get("model_calls"))
            reported_cost += _nonnegative_float(summary.get("reported_cost_usd"))
            _merge_model_spend(model_spend, summary.get("model_spend"))

        for source in _source_runs(current / "transcript.jsonl"):
            resolved = _resolve_source_run(source, root)
            if resolved is None:
                unresolved.add(source)
            elif resolved not in visited:
                pending.append(resolved)

    return {
        "run_count": len(visited),
        "model_calls": model_calls,
        "reported_cost_usd": reported_cost,
        "model_spend": model_spend,
        "runs": [str(path) for path in sorted(visited)],
        "complete": not unresolved and not transcript_fallbacks,
        "unresolved_source_runs": sorted(unresolved),
        "transcript_fallback_runs": sorted(transcript_fallbacks),
    }


def _source_runs(transcript_path: Path) -> set[str]:
    if not transcript_path.exists():
        return set()
    sources = set()
    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            entry = json.loads(line)
            metadata = entry.get("metadata", {}) if isinstance(entry, dict) else {}
            source = metadata.get("source_run") if isinstance(metadata, dict) else None
            if isinstance(source, str) and source.strip():
                sources.add(source)
    except (OSError, json.JSONDecodeError):
        return set()
    return sources


def _resolve_source_run(value: str, root: Path) -> Path | None:
    source = Path(value).expanduser()
    candidates = [source] if source.is_absolute() else [
        Path.cwd() / source,
        root.parent.parent / source,
        root.parent / source.name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _merge_model_spend(target: dict[str, dict[str, Any]], value: Any) -> None:
    if not isinstance(value, dict):
        return
    for model_ref, item in value.items():
        if not isinstance(model_ref, str) or not isinstance(item, dict):
            continue
        aggregate = target.setdefault(
            model_ref,
            {
                "provider": item.get("provider"),
                "provider_model_id": item.get("provider_model_id"),
                "model_calls": 0,
                "reported_cost_usd": 0.0,
            },
        )
        aggregate["model_calls"] += _nonnegative_int(item.get("model_calls"))
        aggregate["reported_cost_usd"] += _nonnegative_float(
            item.get("reported_cost_usd")
        )


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _spend_from_transcript(path: Path) -> dict[str, Any]:
    model_spend: dict[str, dict[str, Any]] = {}
    try:
        entries = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return {"model_calls": 0, "reported_cost_usd": 0.0, "model_spend": {}}
    for entry in entries:
        metadata = entry.get("metadata", {}) if isinstance(entry, dict) else {}
        if not isinstance(metadata, dict) or metadata.get("provider") == "preauthored":
            continue
        model_ref = metadata.get("model_ref")
        if not isinstance(model_ref, str):
            continue
        cost = _nonnegative_float((metadata.get("usage") or {}).get("cost"))
        _merge_model_spend(
            model_spend,
            {
                model_ref: {
                    "provider": metadata.get("provider"),
                    "provider_model_id": metadata.get("model"),
                    "model_calls": 1,
                    "reported_cost_usd": cost,
                }
            },
        )
    return {
        "model_calls": sum(item["model_calls"] for item in model_spend.values()),
        "reported_cost_usd": sum(
            item["reported_cost_usd"] for item in model_spend.values()
        ),
        "model_spend": model_spend,
    }


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _nonnegative_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed >= 0 else 0.0
