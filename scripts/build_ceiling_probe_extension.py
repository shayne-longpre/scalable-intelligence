from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


DEFAULT_GUIDANCE = (
    "Candidates may be substantially less capable, similarly capable, or "
    "substantially more capable than you. Use the remaining probes to add "
    "complementary evidence across that range. Include enough accessible signal "
    "to expose weak systems and enough headroom to distinguish systems beyond "
    "your own capabilities. Do not limit difficulty to problems you are "
    "confident you can solve yourself. Every probe must still be well-posed, "
    "checkable, and answerable without hidden information or external tools."
)


def build_ceiling_extension_config(
    source_run: str | Path,
    *,
    additional_probes: int = 5,
    guidance: str = DEFAULT_GUIDANCE,
    name_suffix: str = "ceiling_extension",
) -> dict[str, Any]:
    if additional_probes < 1:
        raise ValueError("additional_probes must be positive")
    source_run = Path(source_run)
    config = deepcopy(_load_json(source_run / "config.json"))
    phases = config.get("protocol", {}).get("phases", [])
    if len(phases) != 1 or phases[0].get("kind") != "independent_judge_ranking":
        raise ValueError(
            "ceiling extension requires one independent_judge_ranking phase"
        )
    phase = phases[0]
    schedule = phase.get("probe_schedule")
    if not isinstance(schedule, list) or not schedule or int(schedule[0]) < 1:
        raise ValueError("source run must have a non-empty probe_schedule")
    opening_count = int(schedule[0])
    _validate_source_opening(
        source_run,
        opening_count=opening_count,
        participant_ids=[
            str(participant["id"])
            for participant in config.get("participants", [])
        ],
    )
    transcript = str(source_run / "transcript.jsonl")
    phase.update(
        {
            "rounds": 1,
            "probe_schedule": [opening_count + additional_probes],
            "probe_generation_guidance": guidance.strip(),
            "preauthored_probe_file": transcript,
            "preauthored_answer_file": str(source_run),
            "preauthored_evidence_file": transcript,
            "preauthored_ranking_file": None,
            "reuse_unavailable_answers": True,
            "replay_source_targets": False,
            "retry_unavailable_rounds": [],
        }
    )
    config["name"] = f"{config['name']}_{name_suffix}"
    run = config.setdefault("run", {})
    candidate_count = len(config.get("participants", []))
    expected_fresh_calls = (
        additional_probes
        + additional_probes * candidate_count
        + additional_probes
        + 1
    )
    run["max_model_calls"] = expected_fresh_calls + max(
        20, expected_fresh_calls // 3
    )
    metadata = config.setdefault("metadata", {})
    metadata.update(
        {
            "design": (
                f"{opening_count} archived unguided probes plus "
                f"{additional_probes} ceiling-aware probes, followed by one "
                "joint ranking."
            ),
            "ceiling_extension_source_run": str(source_run),
            "archived_opening_probe_count": opening_count,
            "ceiling_aware_probe_count": additional_probes,
            "probe_generation_guidance": guidance.strip(),
        }
    )
    return config


def build_study_configs(
    study_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    study_path = Path(study_path)
    study = _load_json(study_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    additional_probes = int(study.get("additional_probes", 5))
    guidance = str(study.get("probe_generation_guidance", DEFAULT_GUIDANCE))
    rows = []
    for condition in study["conditions"]:
        condition_id = str(condition["id"])
        config = build_ceiling_extension_config(
            condition["source_run"],
            additional_probes=additional_probes,
            guidance=guidance,
            name_suffix=f"{study['name']}_{condition_id}",
        )
        config["metadata"].update(
            {
                "study_file": str(study_path),
                "study_condition": condition_id,
                "research_question": study["research_question"],
            }
        )
        output_path = output_dir / f"{condition_id}.json"
        output_path.write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "condition_id": condition_id,
                "source_run": condition["source_run"],
                "judge_model": _judge_model(config),
                "candidate_count": len(config["participants"]),
                "config": str(output_path),
            }
        )
    index = {
        "study": study["name"],
        "study_file": str(study_path),
        "additional_probes": additional_probes,
        "configs": rows,
    }
    (output_dir / "index.json").write_text(
        json.dumps(index, indent=2) + "\n",
        encoding="utf-8",
    )
    return index


def _judge_model(config: dict[str, Any]) -> str:
    judge_ref = config["judges"][0]["model"]
    models = config["models"]
    if isinstance(models, list):
        model = next(row for row in models if row["name"] == judge_ref)
    else:
        model = models[judge_ref]
    return str(model["model"])


def _validate_source_opening(
    source_run: Path,
    *,
    opening_count: int,
    participant_ids: list[str],
) -> None:
    summary_path = source_run / "run_summary.json"
    if not summary_path.exists() or _load_json(summary_path).get("status") != "completed":
        raise ValueError("ceiling extension source run must be completed")
    turns = [
        json.loads(line)
        for line in (source_run / "transcript.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    opening_probes = {}
    answers: set[tuple[str, str]] = set()
    comparisons: set[str] = set()
    for turn in turns:
        metadata = turn.get("metadata") or {}
        role = metadata.get("interaction_role")
        probe_id = metadata.get("probe_id")
        if role == "question" and turn.get("round_index") == 1:
            number = metadata.get("probe_number")
            if (
                isinstance(number, int)
                and 1 <= number <= opening_count
                and isinstance(probe_id, str)
                and str(turn.get("content", "")).strip()
                and metadata.get("finish_reason") != "length"
            ):
                opening_probes[number] = probe_id
        elif (
            role == "answer"
            and isinstance(probe_id, str)
            and str(turn.get("content", "")).strip()
            and metadata.get("finish_reason") != "length"
            and not metadata.get("answer_unavailable")
        ):
            answers.add((probe_id, str(turn.get("speaker"))))
        elif (
            role == "probe_comparison"
            and isinstance(probe_id, str)
            and str(turn.get("content", "")).strip()
            and metadata.get("finish_reason") != "length"
        ):
            comparisons.add(probe_id)
    expected_numbers = set(range(1, opening_count + 1))
    if set(opening_probes) != expected_numbers:
        raise ValueError(
            "ceiling extension source opening probes are incomplete; "
            f"expected={sorted(expected_numbers)}, "
            f"found={sorted(opening_probes)}"
        )
    missing_answers = [
        (probe_id, participant_id)
        for probe_id in opening_probes.values()
        for participant_id in participant_ids
        if (probe_id, participant_id) not in answers
    ]
    if missing_answers:
        raise ValueError(
            "ceiling extension source opening answers are incomplete; "
            f"missing={missing_answers[:5]}"
        )
    missing_comparisons = [
        probe_id
        for probe_id in opening_probes.values()
        if probe_id not in comparisons
    ]
    if missing_comparisons:
        raise ValueError(
            "ceiling extension source opening comparisons are incomplete; "
            f"missing={missing_comparisons}"
        )


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extend archived opening batteries with fresh ceiling-aware probes "
            "without repeating candidate calls for the original probes."
        )
    )
    parser.add_argument("--study")
    parser.add_argument("--source-run")
    parser.add_argument("--additional-probes", type=int, default=5)
    parser.add_argument("--output-dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    if bool(args.study) == bool(args.source_run):
        parser.error("provide exactly one of --study or --source-run")
    if args.study:
        if not args.output_dir:
            parser.error("--output-dir is required with --study")
        result = build_study_configs(args.study, args.output_dir)
    else:
        if not args.output:
            parser.error("--output is required with --source-run")
        config = build_ceiling_extension_config(
            args.source_run,
            additional_probes=args.additional_probes,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        result = {"output": str(output), "name": config["name"]}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
