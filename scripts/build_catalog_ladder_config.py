from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one anonymous catalog-ladder run config.")
    parser.add_argument("--selection", default="data/catalog_ladder_50.selection.json")
    parser.add_argument("--catalog")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-effort", default="xhigh")
    parser.add_argument("--probe-schedule", default="4,1,1")
    parser.add_argument("--comparison-seed", type=int, default=20260720)
    parser.add_argument("--candidate-timeout-seconds", type=float, default=300)
    parser.add_argument("--judge-timeout-seconds", type=float, default=900)
    parser.add_argument("--model-overrides")
    parser.add_argument(
        "--incomplete-answer-policy",
        choices=("fail", "record_unavailable"),
        default="record_unavailable",
    )
    parser.add_argument("--reuse-unavailable-answers", action="store_true")
    parser.add_argument(
        "--retry-unavailable-rounds",
        help="Comma-separated replay rounds whose unavailable answers should be called again.",
    )
    parser.add_argument("--preauthored-probe-file")
    parser.add_argument("--preauthored-answer-file")
    parser.add_argument("--preauthored-evidence-file")
    parser.add_argument("--preauthored-ranking-file")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selection_path = Path(args.selection)
    selection = _load_json(selection_path)
    catalog_path = Path(args.catalog or selection["catalog"])
    catalog = _load_json(catalog_path)
    catalog_by_id = {model["provider_model_id"]: model for model in catalog["models"]}
    selected_ids = selection["provider_model_ids"]
    model_overrides = _load_json(Path(args.model_overrides)) if args.model_overrides else {}
    if not isinstance(model_overrides, dict):
        raise ValueError("model overrides must be a JSON object keyed by provider route")
    unknown_overrides = sorted(set(model_overrides) - set(selected_ids))
    if unknown_overrides:
        raise ValueError(f"model overrides contain routes outside the selection: {unknown_overrides}")
    missing = [model_id for model_id in selected_ids if model_id not in catalog_by_id]
    if missing:
        raise ValueError(f"selection contains models absent from catalog: {missing}")

    participant_ids = [f"P{index:02d}" for index in range(1, len(selected_ids) + 1)]
    random.Random(int(selection["participant_seed"])).shuffle(participant_ids)
    assignments = dict(zip(selected_ids, participant_ids, strict=True))
    models = [
        _candidate_model(
            catalog_by_id[model_id],
            assignments[model_id],
            model_overrides.get(model_id, {}),
        )
        for model_id in selected_ids
    ]
    judge_ref = "judge_primary"
    models.append(
        {
            "name": judge_ref,
            "provider": "openrouter_judge",
            "model": args.judge_model,
            "params": {
                "max_tokens": 50000,
                "reasoning": {"effort": args.judge_effort},
            },
            "recovery_params": {
                "max_tokens": 30000,
                "reasoning": {"effort": "low"},
            },
        }
    )
    schedule = [int(value) for value in args.probe_schedule.split(",") if value.strip()]
    max_adaptive_candidates = 10
    expected_calls = _expected_model_calls(
        candidate_count=len(selected_ids),
        probe_schedule=schedule,
        max_adaptive_candidates=max_adaptive_candidates,
    )
    phase = {
        "name": "judge_ranking",
        "kind": "independent_judge_ranking",
        "prompt": "adaptive_judge_probe",
        "question_prompt": "adaptive_judge_probe",
        "answer_prompt": "independent_judge_answer",
        "assessment_prompt": "independent_judge_probe_comparison",
        "ranking_prompt": "independent_judge_wave_judgment",
        "probe_schedule": schedule,
        "adaptive_targeting": "judge_selected",
        "max_adaptive_candidates": max_adaptive_candidates,
        "comparison_order": "seeded_shuffle",
        "comparison_seed": args.comparison_seed,
        "incomplete_answer_policy": args.incomplete_answer_policy,
        "reuse_unavailable_answers": args.reuse_unavailable_answers,
        "visibility": "private",
    }
    if args.retry_unavailable_rounds:
        phase["retry_unavailable_rounds"] = [
            int(value)
            for value in args.retry_unavailable_rounds.split(",")
            if value.strip()
        ]
    if args.preauthored_probe_file:
        phase["preauthored_probe_file"] = args.preauthored_probe_file
    if args.preauthored_answer_file:
        phase["preauthored_answer_file"] = args.preauthored_answer_file
    if args.preauthored_evidence_file:
        phase["preauthored_evidence_file"] = args.preauthored_evidence_file
    if args.preauthored_ranking_file:
        phase["preauthored_ranking_file"] = args.preauthored_ranking_file
    config = {
        "name": f"catalog_ladder{len(selected_ids)}_{_safe_name(args.judge_model)}",
        "run": {
            "output_dir": "runs",
            "max_context_turns": 40,
            "max_parallel_calls": 16,
            "max_model_calls": expected_calls + max(20, expected_calls // 5),
            "max_reported_cost_usd": 100.0,
            "structured_json_retries": 2,
            "visible_text_retries": 0,
            "continue_batch_on_call_error": True,
        },
        "context": {
            "mode": "private_memory",
            "max_public_turns": 0,
            "max_private_turns": 12,
            "max_stream_turns": 8,
        },
        "providers": [
            _openrouter_provider(
                "openrouter_candidates",
                args.candidate_timeout_seconds,
                request_retries=0,
            ),
            _openrouter_provider(
                "openrouter_judge",
                args.judge_timeout_seconds,
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
                "model": judge_ref,
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
            "selection_file": str(selection_path),
            "catalog_file": str(catalog_path),
            "catalog_generated_at": catalog.get("generated_at"),
            "participant_seed": selection["participant_seed"],
            "comparison_seed": args.comparison_seed,
            "gold_prior": str(catalog_path),
            **({"model_overrides_file": args.model_overrides} if args.model_overrides else {}),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return 0


def _candidate_model(model: dict, participant_id: str, override: dict) -> dict:
    reasoning_effort = _reasoning_effort(model)
    route_limit = model.get("max_completion_tokens")
    max_tokens = min(40000, int(route_limit)) if route_limit else 40000
    params = {"max_tokens": max_tokens}
    recovery_params = {"max_tokens": 12000 if reasoning_effort else 4000}
    if reasoning_effort:
        params["reasoning"] = {"effort": reasoning_effort}
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
    for field in ("params", "recovery_params"):
        values = override.get(field, {})
        if not isinstance(values, dict):
            raise ValueError(f"model override {field} must be an object")
        result[field].update(values)
    return result


def _reasoning_effort(model: dict) -> str | None:
    variants = " ".join(
        str(item.get("model_name", "")) for item in model.get("matched_evals", [])
    ).lower()
    if any(label in variants for label in ("max effort", "(max)", "xhigh")):
        return "xhigh"
    if "high" in variants or "reasoning" in variants:
        return "high"
    return None


def _expected_model_calls(
    *,
    candidate_count: int,
    probe_schedule: list[int],
    max_adaptive_candidates: int,
) -> int:
    if not probe_schedule or any(count < 1 for count in probe_schedule):
        raise ValueError("probe schedule must contain positive integers")
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


def _openrouter_provider(
    name: str,
    timeout_seconds: float,
    *,
    request_retries: int,
) -> dict:
    return {
        "name": name,
        "kind": "openrouter",
        "api_key_env": "OPENROUTER_API_KEY",
        "headers": {
            "HTTP-Referer": "https://github.com/shayne-longpre/scalable-intelligence",
            "X-Title": "Scalable Intelligence Catalog Ladder",
        },
        "timeout_seconds": timeout_seconds,
        "request_retries": request_retries,
    }


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_name(value: str) -> str:
    return value.rsplit("/", 1)[-1].replace(".", "_").replace("-", "_")


if __name__ == "__main__":
    raise SystemExit(main())
