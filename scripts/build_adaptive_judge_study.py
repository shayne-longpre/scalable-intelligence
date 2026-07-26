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
    catalog_routes = {
        model["provider_model_id"] for model in catalog.get("models", [])
    }
    protocol = study["protocol"]
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
            ),
            selected_model_ids=candidates,
            catalog=catalog,
            catalog_label=str(resolved_catalog_path),
            selection_label=f"{study_path}#{condition_id}",
        )
        config["metadata"]["study_file"] = str(study_path)
        config["metadata"]["study_condition"] = condition_id
        config["metadata"]["research_question"] = study["research_question"]

        config_path = output_dir / f"{condition_id}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        generated.append(
            {
                "condition_id": condition_id,
                "judge_model": judge["model"],
                "config": str(config_path),
                "candidate_count": len(candidates),
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
