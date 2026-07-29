from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def build_partial_resume_config(
    source_run: str | Path,
    *,
    name_suffix: str = "resumed",
    use_judge_recovery_params: bool = False,
    supplement_run: str | Path | None = None,
    bundle_dir: str | Path | None = None,
) -> dict[str, Any]:
    source_run = Path(source_run)
    config = _load_json(source_run / "config.json")
    phases = config.get("protocol", {}).get("phases", [])
    if len(phases) != 1 or phases[0].get("kind") != "independent_judge_ranking":
        raise ValueError(
            "partial resume requires one independent_judge_ranking phase"
        )

    evidence_transcript = str(source_run / "transcript.jsonl")
    replay_source = source_run
    if supplement_run is not None:
        if bundle_dir is None:
            raise ValueError("bundle_dir is required with supplement_run")
        replay_source = write_replay_bundle(
            primary_run=source_run,
            supplement_run=supplement_run,
            output_dir=bundle_dir,
        )
    transcript = str(replay_source / "transcript.jsonl")
    phase = phases[0]
    phase.update(
        {
            "preauthored_probe_file": transcript,
            "preauthored_answer_file": str(replay_source),
            "preauthored_evidence_file": evidence_transcript,
            "preauthored_ranking_file": evidence_transcript,
            "reuse_unavailable_answers": True,
            "replay_source_targets": supplement_run is not None,
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


def write_replay_bundle(
    *,
    primary_run: str | Path,
    supplement_run: str | Path,
    output_dir: str | Path,
) -> Path:
    primary_run = Path(primary_run)
    supplement_run = Path(supplement_run)
    output_dir = Path(output_dir)
    primary = _load_jsonl(primary_run / "transcript.jsonl")
    supplement = _load_jsonl(supplement_run / "transcript.jsonl")
    rows = list(primary)
    identities = {
        identity
        for row in primary
        if (identity := _replay_identity(row)) is not None
    }
    for row in supplement:
        identity = _replay_identity(row)
        if identity is None or identity in identities:
            continue
        rows.append(row)
        identities.add(identity)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transcript.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return output_dir


def _replay_identity(row: Mapping[str, Any]) -> tuple[str, str] | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    role = metadata.get("interaction_role")
    stream_id = metadata.get("stream_id")
    if role not in {"question", "answer"} or not isinstance(stream_id, str):
        return None
    content = row.get("content")
    if role == "answer" and (
        not isinstance(content, str)
        or (
            not content.strip()
            and metadata.get("answer_unavailable") is not True
        )
    ):
        return None
    return role, stream_id


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
    parser.add_argument("--supplement-run")
    parser.add_argument("--bundle-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = build_partial_resume_config(
        args.source_run,
        name_suffix=args.name_suffix,
        use_judge_recovery_params=args.use_judge_recovery_params,
        supplement_run=args.supplement_run,
        bundle_dir=args.bundle_dir,
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
