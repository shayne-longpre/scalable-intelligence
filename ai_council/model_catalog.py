from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import re
from typing import Any


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
ARTIFICIAL_ANALYSIS_MODELS_URL = "https://artificialanalysis.ai/leaderboards/models"
ARENA_LEADERBOARD_URL = "https://arena.ai/leaderboard"


def parse_artificial_analysis_models_html(html: str) -> list[dict[str, Any]]:
    """Extract the model leaderboard payload embedded in the Next.js page."""
    objects = []
    needle = r"{\"models\":["
    offset = 0
    while True:
        start = html.find(needle, offset)
        if start < 0:
            break
        offset = start + 1
        raw_object = _balanced_object(html, start)
        if raw_object is None:
            continue
        try:
            decoded = json.loads(f'"{raw_object}"')
            parsed = json.loads(decoded)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("models"), list):
            objects.append(parsed)

    if not objects:
        raise ValueError("could not find Artificial Analysis model payload")
    best = max(objects, key=lambda item: _intelligence_count(item.get("models", [])))
    return list(best["models"])


def build_openrouter_catalog(
    openrouter_payload: dict[str, Any],
    artificial_analysis_models: list[dict[str, Any]],
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    openrouter_models = list(openrouter_payload.get("data", []))
    aa_index = _build_artificial_analysis_index(artificial_analysis_models)
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()

    entries = []
    for model in openrouter_models:
        provider_model_id = str(model["id"])
        matches = _matches_for_openrouter_model(model, aa_index)
        best_match = _best_match(matches)
        entries.append(_catalog_entry(model, matches, best_match))

    entries.sort(key=_catalog_sort_key)
    for rank, entry in enumerate(entries, start=1):
        entry["estimated_rank"] = rank

    summary = {
        "openrouter_model_count": len(openrouter_models),
        "catalog_model_count": len(entries),
        "artificial_analysis_model_count": len(artificial_analysis_models),
        "ranked_by_reported_score_count": sum(
            1 for entry in entries if entry["ranking_basis"] == "artificial_analysis_reported"
        ),
        "ranked_by_estimated_score_count": sum(
            1 for entry in entries if entry["ranking_basis"] == "artificial_analysis_estimated"
        ),
        "ranked_by_metadata_fallback_count": sum(
            1 for entry in entries if entry["ranking_basis"] == "openrouter_metadata_fallback"
        ),
    }
    return {
        "schema_version": "2026-06-27",
        "name": "openrouter_model_catalog",
        "version": generated_at[:10],
        "generated_at": generated_at,
        "description": (
            "Catalog of OpenRouter-accessible models with best-effort intelligence "
            "ordering. Ranks are based first on Artificial Analysis Intelligence "
            "Index scores and then, only where no score is matched, on weak "
            "OpenRouter metadata fallback ordering."
        ),
        "sources": [
            {
                "name": "OpenRouter Models API",
                "url": OPENROUTER_MODELS_URL,
                "fields_used": [
                    "id",
                    "name",
                    "created",
                    "context_length",
                    "pricing",
                    "architecture",
                    "top_provider",
                    "knowledge_cutoff",
                ],
            },
            {
                "name": "Artificial Analysis LLM Leaderboard",
                "url": ARTIFICIAL_ANALYSIS_MODELS_URL,
                "fields_used": [
                    "Artificial Analysis Intelligence Index",
                    "releaseDate",
                    "creator",
                    "openrouterApiId",
                    "contextWindowTokens",
                    "benchmark component scores",
                ],
            },
            {
                "name": "Arena Leaderboard",
                "url": ARENA_LEADERBOARD_URL,
                "fields_used": [],
                "note": "Referenced as a public leaderboard source; not parsed into this catalog yet.",
            },
        ],
        "ranking_notes": [
            "Lower estimated_rank means stronger expected general capability.",
            "Rows with artificial_analysis_reported are the strongest catalog evidence.",
            "Rows with artificial_analysis_estimated are useful but should be treated as softer priors.",
            "Rows with openrouter_metadata_fallback are access catalog rows, not measured intelligence claims.",
        ],
        "summary": summary,
        "models": entries,
    }


def _catalog_entry(
    model: dict[str, Any],
    matches: list[dict[str, Any]],
    best_match: dict[str, Any] | None,
) -> dict[str, Any]:
    created = model.get("created")
    openrouter_created_date = _date_from_timestamp(created)
    architecture = model.get("architecture") or {}
    top_provider = model.get("top_provider") or {}
    pricing = model.get("pricing") or {}

    if best_match is None:
        ranking_basis = "openrouter_metadata_fallback"
        rank_confidence = "very_low"
        release_date = openrouter_created_date
        intelligence_score = None
        score_is_estimated = None
    else:
        score_is_estimated = bool(best_match.get("intelligenceIndexIsEstimated"))
        ranking_basis = "artificial_analysis_estimated" if score_is_estimated else "artificial_analysis_reported"
        rank_confidence = _rank_confidence(best_match)
        release_date = best_match.get("releaseDate") or openrouter_created_date
        intelligence_score = best_match.get("intelligenceIndex")

    return {
        "provider_model_id": model["id"],
        "display_name": model.get("name") or model["id"],
        "estimated_rank": None,
        "rank_confidence": rank_confidence,
        "ranking_basis": ranking_basis,
        "intelligence_score": intelligence_score,
        "intelligence_score_source": (
            "Artificial Analysis Intelligence Index" if best_match is not None else None
        ),
        "intelligence_score_is_estimated": score_is_estimated,
        "release_date": release_date,
        "release_date_source": "Artificial Analysis" if best_match is not None else "OpenRouter created timestamp",
        "openrouter_created_date": openrouter_created_date,
        "knowledge_cutoff": model.get("knowledge_cutoff"),
        "context_length": model.get("context_length"),
        "max_completion_tokens": top_provider.get("max_completion_tokens"),
        "input_modalities": architecture.get("input_modalities") or [],
        "output_modalities": architecture.get("output_modalities") or [],
        "tokenizer": architecture.get("tokenizer"),
        "openrouter_canonical_slug": model.get("canonical_slug"),
        "hugging_face_id": model.get("hugging_face_id"),
        "pricing": {
            key: pricing.get(key)
            for key in (
                "prompt",
                "completion",
                "input_cache_read",
                "input_cache_write",
                "web_search",
            )
            if key in pricing
        },
        "matched_evals": [_eval_summary(match) for match in matches[:8]],
        "notes": _catalog_note(best_match),
    }


def _build_artificial_analysis_index(
    models: list[dict[str, Any]]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    by_openrouter_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_slug_key: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for model in models:
        if model.get("intelligenceIndex") is None:
            continue
        openrouter_id = model.get("openrouterApiId")
        if isinstance(openrouter_id, str) and openrouter_id:
            by_openrouter_id[openrouter_id].append({**model, "_match_type": "openrouter_api_id"})
        for value in (model.get("slug"), model.get("shortName"), model.get("name")):
            key = _normalize_key(value)
            if key:
                by_slug_key[key].append({**model, "_match_type": "slug_or_name"})

    return {
        "by_openrouter_id": dict(by_openrouter_id),
        "by_slug_key": dict(by_slug_key),
    }


def _matches_for_openrouter_model(
    model: dict[str, Any],
    aa_index: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    model_id = str(model["id"])
    matches = list(aa_index["by_openrouter_id"].get(model_id, []))

    if matches:
        return sorted(matches, key=_match_sort_key)

    keys = {
        _normalize_key(model_id.rsplit("/", 1)[-1]),
        _normalize_key(model.get("canonical_slug")),
        _normalize_key((model.get("name") or "").split(":", 1)[-1]),
    }
    slug_matches = []
    for key in keys:
        if not key:
            continue
        slug_matches.extend(aa_index["by_slug_key"].get(key, []))
    deduped = {_eval_identity(match): match for match in slug_matches}
    return sorted(deduped.values(), key=_match_sort_key)


def _best_match(matches: list[dict[str, Any]]) -> dict[str, Any] | None:
    return matches[0] if matches else None


def _match_sort_key(match: dict[str, Any]) -> tuple[int, float, str]:
    score = float(match.get("intelligenceIndex") or -999.0)
    estimated_penalty = 1 if match.get("intelligenceIndexIsEstimated") else 0
    return (estimated_penalty, -score, str(match.get("name") or ""))


def _catalog_sort_key(entry: dict[str, Any]) -> tuple[int, float, str, str]:
    score = entry.get("intelligence_score")
    if score is not None:
        confidence_bucket = {
            "high": 0,
            "medium": 1,
            "low": 2,
        }.get(entry.get("rank_confidence"), 3)
        return (0, -float(score), str(confidence_bucket), entry["provider_model_id"])
    return (1, -_fallback_recency(entry), "9", entry["provider_model_id"])


def _fallback_recency(entry: dict[str, Any]) -> float:
    date = entry.get("openrouter_created_date") or ""
    digits = re.sub(r"\D", "", date)
    return float(digits or 0)


def _eval_summary(match: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "Artificial Analysis Intelligence Index",
        "match_type": match.get("_match_type"),
        "model_name": match.get("name"),
        "slug": match.get("slug"),
        "score": match.get("intelligenceIndex"),
        "score_is_estimated": match.get("intelligenceIndexIsEstimated"),
        "release_date": match.get("releaseDate"),
        "deprecated": match.get("deprecated"),
        "creator": match.get("modelCreatorName"),
        "openrouter_api_id": match.get("openrouterApiId"),
        "coding_index": match.get("codingIndex"),
        "agentic_index": match.get("agenticIndex"),
        "hle": match.get("hle"),
        "gpqa": match.get("gpqa"),
        "ifbench": match.get("ifbench"),
    }


def _rank_confidence(match: dict[str, Any]) -> str:
    exact = match.get("_match_type") == "openrouter_api_id"
    estimated = bool(match.get("intelligenceIndexIsEstimated"))
    if exact and not estimated:
        return "high"
    if exact or not estimated:
        return "medium"
    return "low"


def _catalog_note(best_match: dict[str, Any] | None) -> str:
    if best_match is None:
        return "No matched reported intelligence score; rank is metadata fallback only."
    if best_match.get("deprecated"):
        return "Best matched benchmark row is marked deprecated by Artificial Analysis."
    if best_match.get("intelligenceIndexIsEstimated"):
        return "Artificial Analysis marks this intelligence score as estimated."
    return "Ranked from matched Artificial Analysis Intelligence Index score."


def _balanced_object(text: str, start: int) -> str | None:
    depth = 0
    for index, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _intelligence_count(models: list[dict[str, Any]]) -> int:
    return sum(1 for model in models if model.get("intelligenceIndex") is not None)


def _eval_identity(match: dict[str, Any]) -> str:
    return str(match.get("id") or match.get("slug") or match.get("name"))


def _normalize_key(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _date_from_timestamp(value: Any) -> str | None:
    if not isinstance(value, int | float):
        return None
    return datetime.fromtimestamp(value, timezone.utc).date().isoformat()
