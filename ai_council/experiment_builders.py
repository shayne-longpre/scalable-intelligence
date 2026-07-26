from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import random


@dataclass(frozen=True)
class AdaptiveJudgeConfigSpec:
    name: str
    judge_model: str
    participant_seed: int
    comparison_seed: int
    probe_schedule: tuple[int, ...] = (4, 1, 1)
    max_adaptive_candidates: int = 10
    judge_params: Mapping[str, Any] | None = None
    judge_recovery_params: Mapping[str, Any] | None = None
    candidate_timeout_seconds: float = 300
    judge_timeout_seconds: float = 900
    candidate_request_retries: int = 0
    incomplete_answer_policy: str = "record_unavailable"
    visible_text_retries: int = 1
    reuse_unavailable_answers: bool = False
    replay_source_targets: bool = False
    retry_unavailable_rounds: tuple[int, ...] = ()
    preauthored_files: Mapping[str, str] = field(default_factory=dict)


def build_adaptive_judge_config(
    *,
    spec: AdaptiveJudgeConfigSpec,
    selected_model_ids: Sequence[str],
    catalog: Mapping[str, Any],
    catalog_label: str,
    selection_label: str,
    model_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    model_overrides_label: str | None = None,
) -> dict[str, Any]:
    selected_ids = list(selected_model_ids)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("candidate routes must be non-empty and unique")

    catalog_by_id = {
        model["provider_model_id"]: model for model in catalog.get("models", [])
    }
    missing = [model_id for model_id in selected_ids if model_id not in catalog_by_id]
    if missing:
        raise ValueError(f"candidate routes absent from catalog: {missing}")
    if spec.judge_model not in catalog_by_id:
        raise ValueError(f"judge route absent from catalog: {spec.judge_model}")

    overrides = dict(model_overrides or {})
    unknown_overrides = sorted(set(overrides) - set(selected_ids))
    if unknown_overrides:
        raise ValueError(
            f"model overrides contain routes outside the candidate panel: {unknown_overrides}"
        )

    participant_ids = [f"P{index:02d}" for index in range(1, len(selected_ids) + 1)]
    random.Random(spec.participant_seed).shuffle(participant_ids)
    assignments = dict(zip(selected_ids, participant_ids, strict=True))
    models = [
        candidate_model(
            catalog_by_id[model_id],
            assignments[model_id],
            overrides.get(model_id, {}),
        )
        for model_id in selected_ids
    ]
    judge_params, judge_recovery_params = judge_model_params(
        catalog_by_id[spec.judge_model],
        params=spec.judge_params,
        recovery_params=spec.judge_recovery_params,
    )
    models.append(
        {
            "name": "judge_primary",
            "provider": "openrouter_judge",
            "model": spec.judge_model,
            "params": judge_params,
            "recovery_params": judge_recovery_params,
        }
    )

    schedule = list(spec.probe_schedule)
    expected_calls = expected_model_calls(
        candidate_count=len(selected_ids),
        probe_schedule=schedule,
        max_adaptive_candidates=spec.max_adaptive_candidates,
    )

    phase: dict[str, Any] = {
        "name": "judge_ranking",
        "kind": "independent_judge_ranking",
        "prompt": "adaptive_judge_probe",
        "question_prompt": "adaptive_judge_probe",
        "answer_prompt": "independent_judge_answer",
        "assessment_prompt": "independent_judge_probe_comparison",
        "ranking_prompt": "independent_judge_wave_judgment",
        "probe_schedule": schedule,
        "adaptive_targeting": "judge_selected",
        "max_adaptive_candidates": spec.max_adaptive_candidates,
        "comparison_order": "seeded_shuffle",
        "comparison_seed": spec.comparison_seed,
        "incomplete_answer_policy": spec.incomplete_answer_policy,
        "reuse_unavailable_answers": spec.reuse_unavailable_answers,
        "visibility": "private",
    }
    if spec.replay_source_targets:
        phase["replay_source_targets"] = True
    if spec.retry_unavailable_rounds:
        phase["retry_unavailable_rounds"] = list(spec.retry_unavailable_rounds)
    for key, path in spec.preauthored_files.items():
        if key not in {
            "preauthored_probe_file",
            "preauthored_answer_file",
            "preauthored_evidence_file",
            "preauthored_ranking_file",
        }:
            raise ValueError(f"unsupported preauthored file field: {key}")
        phase[key] = path

    return {
        "name": spec.name,
        "run": {
            "output_dir": "runs",
            "max_context_turns": 40,
            "max_parallel_calls": 16,
            "max_model_calls": expected_calls + max(20, expected_calls // 5),
            "max_reported_cost_usd": 100.0,
            "structured_json_retries": 2,
            "visible_text_retries": spec.visible_text_retries,
            "continue_batch_on_call_error": True,
        },
        "context": {
            "mode": "private_memory",
            "max_public_turns": 0,
            "max_private_turns": 12,
            "max_stream_turns": 8,
        },
        "providers": [
            openrouter_provider(
                "openrouter_candidates",
                spec.candidate_timeout_seconds,
                request_retries=spec.candidate_request_retries,
            ),
            openrouter_provider(
                "openrouter_judge",
                spec.judge_timeout_seconds,
                request_retries=1,
            ),
        ],
        "models": models,
        "participants": [
            {
                "id": assignments[model_id],
                "model": f"candidate_{assignments[model_id].lower()}",
                "system_prompt": "blind_evaluation_candidate",
            }
            for model_id in selected_ids
        ],
        "judges": [
            {
                "id": "J1",
                "model": "judge_primary",
                "system_prompt": "independent_intelligence_judge",
            }
        ],
        "monitor": {"enabled": True, "strict": False},
        "protocol": {
            "name": "catalog_ladder_adaptive_waves",
            "turn_order": "fixed",
            "phases": [phase],
        },
        "metadata": {
            "design": "Opening probe portfolio followed by selective adaptive tie-breaking.",
            "selection_file": selection_label,
            "catalog_file": catalog_label,
            "catalog_generated_at": catalog.get("generated_at"),
            "participant_seed": spec.participant_seed,
            "comparison_seed": spec.comparison_seed,
            "gold_prior": catalog_label,
            **(
                {"model_overrides_file": model_overrides_label}
                if model_overrides_label
                else {}
            ),
        },
    }


def candidate_model(
    model: Mapping[str, Any],
    participant_id: str,
    override: Mapping[str, Any],
) -> dict[str, Any]:
    effort = reasoning_effort(model)
    route_limit = model.get("max_completion_tokens")
    max_tokens = min(40000, int(route_limit)) if route_limit else 40000
    params: dict[str, Any] = {"max_tokens": max_tokens}
    recovery_params: dict[str, Any] = {
        "max_tokens": 12000 if effort else 4000
    }
    if effort:
        params["reasoning"] = {"effort": effort}
        recovery_params["reasoning"] = {"effort": "low"}
    else:
        params["temperature"] = 0.2
        recovery_params["temperature"] = 0
    result = {
        "name": f"candidate_{participant_id.lower()}",
        "provider": "openrouter_candidates",
        "model": model["provider_model_id"],
        "params": params,
        "recovery_params": recovery_params,
    }
    for field_name in ("params", "recovery_params"):
        values = override.get(field_name, {})
        if not isinstance(values, Mapping):
            raise ValueError(f"model override {field_name} must be an object")
        result[field_name].update(values)
    return result


def judge_model_params(
    model: Mapping[str, Any],
    *,
    params: Mapping[str, Any] | None = None,
    recovery_params: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    effort = reasoning_effort(model)
    route_limit = model.get("max_completion_tokens")
    max_tokens = min(50000, int(route_limit)) if route_limit else 50000
    if effort:
        defaults: dict[str, Any] = {
            "max_tokens": max_tokens,
            "reasoning": {"effort": effort},
        }
        recovery_defaults: dict[str, Any] = {
            "max_tokens": min(30000, max_tokens),
            "reasoning": {"effort": "low"},
        }
    else:
        defaults = {"max_tokens": min(20000, max_tokens), "temperature": 0.2}
        recovery_defaults = {
            "max_tokens": min(8000, max_tokens),
            "temperature": 0,
        }
    defaults.update(params or {})
    recovery_defaults.update(recovery_params or {})
    return defaults, recovery_defaults


def reasoning_effort(model: Mapping[str, Any]) -> str | None:
    variants = " ".join(
        str(item.get("model_name", "")) for item in model.get("matched_evals", [])
    ).lower()
    if any(label in variants for label in ("max effort", "(max)", "xhigh")):
        return "xhigh"
    if "high" in variants or "reasoning" in variants:
        return "high"
    return None


def expected_model_calls(
    *,
    candidate_count: int,
    probe_schedule: Sequence[int],
    max_adaptive_candidates: int,
) -> int:
    if candidate_count < 1:
        raise ValueError("candidate count must be positive")
    if not probe_schedule or any(count < 1 for count in probe_schedule):
        raise ValueError("probe schedule must contain positive integers")
    if max_adaptive_candidates < 1 or max_adaptive_candidates > candidate_count:
        raise ValueError(
            "max adaptive candidates must be between one and the candidate count"
        )
    opening_answers = candidate_count * probe_schedule[0]
    adaptive_answers = max_adaptive_candidates * sum(probe_schedule[1:])
    probe_authoring_and_comparison = 2 * sum(probe_schedule)
    cumulative_judgments = len(probe_schedule)
    return (
        opening_answers
        + adaptive_answers
        + probe_authoring_and_comparison
        + cumulative_judgments
    )


def openrouter_provider(
    name: str,
    timeout_seconds: float,
    *,
    request_retries: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "openrouter",
        "api_key_env": "OPENROUTER_API_KEY",
        "headers": {
            "HTTP-Referer": "https://github.com/shayne-longpre/scalable-intelligence",
            "X-Title": "Scalable Intelligence",
        },
        "timeout_seconds": timeout_seconds,
        "request_retries": request_retries,
    }


def build_exact_evidence_order_replay_config(
    source_config: Mapping[str, Any],
    *,
    source_run: str | Path,
    comparison_seed: int,
    study_condition: str,
    name_suffix: str = "order_replay",
) -> dict[str, Any]:
    config = deepcopy(dict(source_config))
    phases = config.get("protocol", {}).get("phases", [])
    if len(phases) != 1 or phases[0].get("kind") != "independent_judge_ranking":
        raise ValueError(
            "order replay requires one independent_judge_ranking phase"
        )
    phase = phases[0]
    source_seed = int(phase.get("comparison_seed", 0))
    if comparison_seed == source_seed:
        raise ValueError("order replay comparison seed must differ from the source")

    source_run = Path(source_run)
    phase.update(
        {
            "comparison_order": "seeded_shuffle",
            "comparison_seed": int(comparison_seed),
            "preauthored_probe_file": str(source_run / "transcript.jsonl"),
            "preauthored_answer_file": str(source_run),
            "reuse_unavailable_answers": True,
            "replay_source_targets": True,
            "retry_unavailable_rounds": [],
        }
    )
    phase.pop("preauthored_evidence_file", None)
    phase.pop("preauthored_ranking_file", None)

    config["name"] = f"{config['name']}_{name_suffix}"
    metadata = config.setdefault("metadata", {})
    source_condition = metadata.get("study_condition")
    metadata.pop("repair_source_run", None)
    metadata.pop("repair_retry_unavailable_rounds", None)
    metadata.update(
        {
            "study_condition": study_condition,
            "source_study_condition": source_condition,
            "order_replay_source_run": str(source_run),
            "order_replay_source_comparison_seed": source_seed,
            "comparison_seed": int(comparison_seed),
        }
    )
    return config
