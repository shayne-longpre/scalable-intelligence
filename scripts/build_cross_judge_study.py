from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_council.experiment_builders import (
    build_exact_evidence_cross_judge_config,
)


def build_cross_judge_study_configs(
    study_path: Path,
    output_dir: Path,
    *,
    condition_ids: set[str] | None = None,
) -> dict[str, Any]:
    study = _load_json(study_path)
    catalog_path = Path(study["catalog"])
    catalog = _load_json(catalog_path)
    known_ids = {condition["id"] for condition in study["conditions"]}
    unknown = sorted((condition_ids or set()) - known_ids)
    if unknown:
        raise ValueError(f"crossed study has no conditions: {unknown}")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for condition in study["conditions"]:
        if condition_ids and condition["id"] not in condition_ids:
            continue
        source_run = Path(condition["source_run"])
        validate_exact_evidence_source(
            source_run,
            condition_id=condition["id"],
            allow_unavailable=bool(
                condition.get(
                    "allow_unavailable_evidence",
                    study.get("allow_unavailable_evidence", False),
                )
            ),
        )
        evaluator = condition["evaluator_model"]
        config = build_exact_evidence_cross_judge_config(
            _load_json(source_run / "config.json"),
            source_run=source_run,
            comparison_seed=int(condition["comparison_seed"]),
            study_condition=condition["id"],
            judge_model=evaluator,
            catalog=catalog,
        )
        config["name"] = f"{study['name']}_{condition['id']}"
        config["metadata"]["study_file"] = str(study_path)
        config["metadata"]["research_question"] = study["research_question"]
        config["metadata"]["probe_author_model"] = condition[
            "probe_author_model"
        ]
        config_path = output_dir / f"{condition['id']}.json"
        config_path.write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )
        generated.append(
            {
                "condition_id": condition["id"],
                "source_run": str(source_run),
                "probe_author_model": condition["probe_author_model"],
                "evaluator_model": evaluator,
                "config": str(config_path),
            }
        )
    index = {
        "study": study["name"],
        "study_file": str(study_path),
        "catalog": str(catalog_path),
        "configs": generated,
    }
    (output_dir / "index.json").write_text(
        json.dumps(index, indent=2) + "\n",
        encoding="utf-8",
    )
    return index


def validate_exact_evidence_source(
    source_run: Path,
    *,
    condition_id: str,
    allow_unavailable: bool = False,
) -> None:
    for filename in ("config.json", "transcript.jsonl", "run_summary.json"):
        if not (source_run / filename).exists():
            raise ValueError(
                f"{condition_id} source run has no {filename}: {source_run}"
            )
    summary = _load_json(source_run / "run_summary.json")
    if summary.get("status") != "completed":
        raise ValueError(f"{condition_id} source run is not completed")
    unavailable = 0
    with (source_run / "transcript.jsonl").open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{condition_id} source transcript has invalid JSON "
                    f"at line {line_number}"
                ) from exc
            unavailable += bool(
                row.get("metadata", {}).get("answer_unavailable")
            )
    if unavailable and not allow_unavailable:
        raise ValueError(
            f"{condition_id} source run has {unavailable} unavailable "
            "candidate answer(s)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build exact-evidence configs with a different evaluator."
    )
    parser.add_argument("--study", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--condition",
        action="append",
        dest="conditions",
        help="Build only this condition ID. May be repeated.",
    )
    args = parser.parse_args()
    index = build_cross_judge_study_configs(
        Path(args.study),
        Path(args.output_dir),
        condition_ids=set(args.conditions) if args.conditions else None,
    )
    print(json.dumps(index, indent=2))
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
