from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_council.experiment_builders import (
    AdaptiveJudgeConfigSpec,
    build_adaptive_judge_config,
)


def build_study_configs(
    study_path: Path,
    output_dir: Path,
    *,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    study = _load_json(study_path)
    resolved_catalog_path = catalog_path or Path(study["catalog"])
    catalog = _load_json(resolved_catalog_path)
    catalog_by_route = {
        model["provider_model_id"]: model for model in catalog.get("models", [])
    }
    catalog_routes = set(catalog_by_route)
    protocol = study["protocol"]
    relative_gap_requirements = study.get("panel_design", {}).get(
        "relative_gap_requirements", []
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []
    for condition in study["conditions"]:
        condition_id = condition["id"]
        candidates = condition["candidate_models"]
        judge = condition["judge"]
        unknown = sorted((set(candidates) | {judge["model"]}) - catalog_routes)
        if unknown:
            raise ValueError(f"{condition_id} contains routes absent from catalog: {unknown}")
        if judge["model"] not in candidates:
            raise ValueError(f"{condition_id} must include the judge in its candidate panel")
        relative_gap_counts = validate_relative_gap_requirements(
            condition_id=condition_id,
            judge_model=judge["model"],
            candidate_models=candidates,
            catalog_by_route=catalog_by_route,
            requirements=relative_gap_requirements,
        )

        config_name = f"{study['name']}_{condition_id}"
        config = build_adaptive_judge_config(
            spec=AdaptiveJudgeConfigSpec(
                name=config_name,
                judge_model=judge["model"],
                judge_params=judge.get("params"),
                judge_recovery_params=judge.get("recovery_params"),
                participant_seed=int(condition["participant_seed"]),
                comparison_seed=int(condition["comparison_seed"]),
                probe_schedule=tuple(protocol["probe_schedule"]),
                max_adaptive_candidates=int(protocol["max_adaptive_candidates"]),
                candidate_timeout_seconds=float(
                    protocol.get("candidate_timeout_seconds", 300)
                ),
                judge_timeout_seconds=float(
                    protocol.get("judge_timeout_seconds", 900)
                ),
                candidate_request_retries=int(
                    protocol.get("candidate_request_retries", 0)
                ),
                incomplete_answer_policy=protocol.get(
                    "incomplete_answer_policy", "record_unavailable"
                ),
                probe_generation_guidance=str(
                    protocol.get("probe_generation_guidance", "")
                ),
            ),
            selected_model_ids=candidates,
            catalog=catalog,
            catalog_label=str(resolved_catalog_path),
            selection_label=f"{study_path}#{condition_id}",
        )
        config["metadata"]["study_file"] = str(study_path)
        config["metadata"]["study_condition"] = condition_id
        config["metadata"]["research_question"] = study["research_question"]
        if protocol.get("primary_endpoint"):
            config["metadata"]["primary_endpoint"] = protocol[
                "primary_endpoint"
            ]

        config_path = output_dir / f"{condition_id}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        generated.append(
            {
                "condition_id": condition_id,
                "judge_model": judge["model"],
                "config": str(config_path),
                "candidate_count": len(candidates),
                "relative_gap_counts": relative_gap_counts,
            }
        )

    index = {
        "study": study["name"],
        "study_file": str(study_path),
        "catalog": str(resolved_catalog_path),
        "configs": generated,
    }
    (output_dir / "index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    return index


def validate_relative_gap_requirements(
    *,
    condition_id: str,
    judge_model: str,
    candidate_models: list[str],
    catalog_by_route: dict[str, dict[str, Any]],
    requirements: list[dict[str, Any]],
) -> dict[str, int]:
    if not requirements:
        return {}
    judge_score = catalog_by_route[judge_model].get("intelligence_score")
    if not isinstance(judge_score, (int, float)):
        raise ValueError(f"{condition_id} judge has no intelligence score")
    deltas = []
    for model in candidate_models:
        if model == judge_model:
            continue
        score = catalog_by_route[model].get("intelligence_score")
        if not isinstance(score, (int, float)):
            raise ValueError(f"{condition_id} candidate {model} has no intelligence score")
        deltas.append(float(score) - float(judge_score))

    counts: dict[str, int] = {}
    for requirement in requirements:
        label = str(requirement["label"])
        if label in counts:
            raise ValueError(
                f"{condition_id} repeats gap requirement label {label}"
            )
        side = requirement["side"]
        if side not in {"above", "below"}:
            raise ValueError(f"{condition_id} gap requirement {label} has invalid side")
        minimum = requirement["minimum"]
        if isinstance(minimum, bool) or not isinstance(minimum, int):
            raise ValueError(
                f"{condition_id} gap requirement {label} minimum must be an integer"
            )
        min_gap = float(requirement.get("min_gap", 0))
        max_gap = requirement.get("max_gap")
        max_gap = float(max_gap) if max_gap is not None else None
        if minimum < 0 or min_gap < 0 or (max_gap is not None and max_gap <= min_gap):
            raise ValueError(f"{condition_id} gap requirement {label} has invalid bounds")
        count = sum(
            _gap_in_requirement(
                delta,
                side=side,
                min_gap=min_gap,
                max_gap=max_gap,
            )
            for delta in deltas
        )
        counts[label] = count
        if count < minimum:
            raise ValueError(
                f"{condition_id} requires {minimum} {label} candidate(s), found {count}"
            )
    return counts


def _gap_in_requirement(
    delta: float,
    *,
    side: str,
    min_gap: float,
    max_gap: float | None,
) -> bool:
    if (side == "above" and delta <= 0) or (side == "below" and delta >= 0):
        return False
    gap = abs(delta)
    return gap >= min_gap and (max_gap is None or gap < max_gap)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build resolved configs for an adaptive-judge study manifest."
    )
    parser.add_argument("--study", required=True)
    parser.add_argument("--catalog")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    index = build_study_configs(
        Path(args.study),
        Path(args.output_dir),
        catalog_path=Path(args.catalog) if args.catalog else None,
    )
    print(json.dumps(index, indent=2))
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
