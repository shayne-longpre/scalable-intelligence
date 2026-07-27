from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_partial_resume_config(
    source_run: str | Path,
    *,
    name_suffix: str = "resumed",
    use_judge_recovery_params: bool = False,
) -> dict[str, Any]:
    source_run = Path(source_run)
    config = _load_json(source_run / "config.json")
    phases = config.get("protocol", {}).get("phases", [])
    if len(phases) != 1 or phases[0].get("kind") != "independent_judge_ranking":
        raise ValueError(
            "partial resume requires one independent_judge_ranking phase"
        )

    transcript = str(source_run / "transcript.jsonl")
    phase = phases[0]
    phase.update(
        {
            "preauthored_probe_file": transcript,
            "preauthored_answer_file": str(source_run),
            "preauthored_evidence_file": transcript,
            "preauthored_ranking_file": transcript,
            "reuse_unavailable_answers": True,
            "replay_source_targets": False,
            "retry_unavailable_rounds": [],
        }
    )
    config["name"] = f"{config['name']}_{name_suffix}"
    metadata = config.setdefault("metadata", {})
    metadata["resume_source_run"] = str(source_run)
    if use_judge_recovery_params:
        _apply_judge_recovery_params(config)
        metadata["resume_judge_uses_recovery_params"] = True
    return config


def _apply_judge_recovery_params(config: dict[str, Any]) -> None:
    model_rows = config.get("models", [])
    if isinstance(model_rows, dict):
        models = model_rows
    else:
        models = {
            str(row.get("name")): row
            for row in model_rows
            if isinstance(row, dict)
        }
    for judge in config.get("judges", []):
        model_ref = str(judge.get("model"))
        model = models.get(model_ref)
        if not isinstance(model, dict):
            raise ValueError(f"partial resume cannot resolve judge model {model_ref}")
        recovery = model.get("recovery_params")
        if not isinstance(recovery, dict):
            raise ValueError(
                f"partial resume judge model {model_ref} has no recovery params"
            )
        model["params"] = dict(recovery)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a config that resumes missing stages from a partial run."
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--name-suffix", default="resumed")
    parser.add_argument("--use-judge-recovery-params", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = build_partial_resume_config(
        args.source_run,
        name_suffix=args.name_suffix,
        use_judge_recovery_params=args.use_judge_recovery_params,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), "name": config["name"]}, indent=2))
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
