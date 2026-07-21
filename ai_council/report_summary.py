from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from ai_council.clients.registry import build_client
from ai_council.config import ConfigError, ProviderSpec
from ai_council.core import ModelRequest


@dataclass(frozen=True)
class ReportSummarySpec:
    provider: ProviderSpec
    model: str
    params: dict[str, Any] = field(default_factory=dict)


def generate_report_summary(
    payload: dict[str, Any],
    *,
    config_path: str | Path,
) -> dict[str, Any]:
    spec = load_report_summary_spec(config_path)
    client = build_client(spec.provider)
    response = client.generate(
        ModelRequest(
            model=spec.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write concise research notes about AI evaluation "
                        "experiments. Be concrete, do not overclaim, and focus on "
                        "what was behaviorally interesting. Treat the supplied "
                        "JSON as the only source of truth."
                    ),
                },
                {"role": "user", "content": _summary_prompt(payload)},
            ],
            params=spec.params,
            metadata={"report_summary": True},
        )
    )
    return {
        "provider": spec.provider.name,
        "model": response.model or spec.model,
        "content": response.content.strip(),
        "usage": response.usage,
    }


def load_report_summary_spec(path: str | Path) -> ReportSummarySpec:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ConfigError("report summary config must be a JSON object")
    provider_data = data.get("provider")
    if not isinstance(provider_data, dict):
        raise ConfigError("report summary config must include provider object")
    model = data.get("model")
    if not isinstance(model, str) or not model:
        raise ConfigError("report summary config must include non-empty model string")
    params = data.get("params", {})
    if not isinstance(params, dict):
        raise ConfigError("report summary params must be an object")
    return ReportSummarySpec(
        provider=ProviderSpec.from_dict(provider_data),
        model=model,
        params=params,
    )


def _summary_prompt(payload: dict[str, Any]) -> str:
    compact = {
        "task": (
            "Write a short highlights summary for a public-facing report card. "
            "Use 2-4 bullets, each under 35 words. Describe one to three concrete "
            "probe tasks and how later probes responded to unresolved evidence. "
            "Mention saturation or ambiguity only when a comparison records it. "
            "Do not merely repeat taxonomy labels or operational recovery events. "
            "Do not mention final rankings, agreement metrics, or winner claims; "
            "deterministic tables elsewhere cover those outcomes."
        ),
        "paired_comparison": payload.get("paired_comparison", {}),
        "runs": [_compact_run(card) for card in payload.get("runs", [])],
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _compact_run(card: dict[str, Any]) -> dict[str, Any]:
    taxonomy = card.get("taxonomy_counts", {})
    return {
        "name": card.get("name"),
        "mode": card.get("mode"),
        "structure": card.get("structure"),
        "prior_expected_order": card.get("prior_expected_order"),
        "final_rankings": card.get("final_rankings"),
        "final_agreement": card.get("final_agreement"),
        "final_prior_agreement": card.get("final_prior_agreement"),
        "question_type_frequency": (
            taxonomy.get("question_type_frequency", {})
            if isinstance(taxonomy, dict)
            else {}
        ),
        "strategy_frequency": (
            taxonomy.get("signal_frequency", {})
            if isinstance(taxonomy, dict)
            else {}
        ),
        "transition_counts": card.get("transition_counts", {}),
        "quality_gates": card.get("quality_gates", {}),
        "question_types_by_round": card.get("question_types_by_round", {}),
        "highlights": card.get("highlights", [])[:5],
        "probe_timeline": [
            {
                "round_index": item.get("round_index"),
                "transition": item.get("transition_label"),
                "excerpt": item.get("excerpt"),
            }
            for item in card.get("event_timeline", [])
            if isinstance(item, dict)
        ][:12],
        "probe_comparisons": [
            {
                "round_index": item.get("round_index"),
                "probe_sequence_number": item.get("probe_sequence_number"),
                "probe_validity": (item.get("parsed") or {}).get("probe_validity"),
                "uncertainties": (item.get("parsed") or {}).get("uncertainties", [])[:2],
            }
            for item in card.get("probe_comparisons", [])
            if isinstance(item, dict) and isinstance(item.get("parsed"), dict)
        ][:12],
    }
