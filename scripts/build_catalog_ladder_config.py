from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_council.experiment_builders import (
    AdaptiveJudgeConfigSpec,
    build_adaptive_judge_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one anonymous catalog-ladder config.")
    parser.add_argument("--selection", default="data/catalog_ladder_50.selection.json")
    parser.add_argument("--catalog")
    parser.add_argument("--name")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-effort")
    parser.add_argument("--probe-schedule", default="4,1,1")
    parser.add_argument("--max-adaptive-candidates", type=int, default=10)
    parser.add_argument("--comparison-seed", type=int, default=20260720)
    parser.add_argument("--participant-seed", type=int)
    parser.add_argument("--candidate-timeout-seconds", type=float, default=300)
    parser.add_argument("--judge-timeout-seconds", type=float, default=900)
    parser.add_argument("--model-overrides")
    parser.add_argument(
        "--incomplete-answer-policy",
        choices=("fail", "record_unavailable"),
        default="record_unavailable",
    )
    parser.add_argument("--reuse-unavailable-answers", action="store_true")
    parser.add_argument("--replay-source-targets", action="store_true")
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
    model_overrides = (
        _load_json(Path(args.model_overrides)) if args.model_overrides else {}
    )
    if not isinstance(model_overrides, dict):
        raise ValueError("model overrides must be a JSON object keyed by provider route")

    participant_seed = (
        args.participant_seed
        if args.participant_seed is not None
        else int(selection["participant_seed"])
    )
    judge_params: dict[str, Any] | None = None
    if args.judge_effort:
        judge_params = {
            "max_tokens": 50000,
            "reasoning": {"effort": args.judge_effort},
        }
    preauthored_files = {
        key: value
        for key, value in {
            "preauthored_probe_file": args.preauthored_probe_file,
            "preauthored_answer_file": args.preauthored_answer_file,
            "preauthored_evidence_file": args.preauthored_evidence_file,
            "preauthored_ranking_file": args.preauthored_ranking_file,
        }.items()
        if value
    }
    spec = AdaptiveJudgeConfigSpec(
        name=args.name
        or f"catalog_ladder{len(selection['provider_model_ids'])}_{_safe_name(args.judge_model)}",
        judge_model=args.judge_model,
        judge_params=judge_params,
        participant_seed=participant_seed,
        comparison_seed=args.comparison_seed,
        probe_schedule=_csv_ints(args.probe_schedule),
        max_adaptive_candidates=args.max_adaptive_candidates,
        candidate_timeout_seconds=args.candidate_timeout_seconds,
        judge_timeout_seconds=args.judge_timeout_seconds,
        incomplete_answer_policy=args.incomplete_answer_policy,
        reuse_unavailable_answers=args.reuse_unavailable_answers,
        replay_source_targets=args.replay_source_targets,
        retry_unavailable_rounds=_csv_ints(args.retry_unavailable_rounds),
        preauthored_files=preauthored_files,
    )
    config = build_adaptive_judge_config(
        spec=spec,
        selected_model_ids=selection["provider_model_ids"],
        catalog=catalog,
        catalog_label=str(catalog_path),
        selection_label=str(selection_path),
        model_overrides=model_overrides,
        model_overrides_label=args.model_overrides,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return 0


def _csv_ints(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(item) for item in value.split(",") if item.strip())


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_name(value: str) -> str:
    return value.rsplit("/", 1)[-1].replace(".", "_").replace("-", "_")


if __name__ == "__main__":
    raise SystemExit(main())
