from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def build_replay_repair_config(
    source_run: str | Path,
    *,
    retry_unavailable_rounds: Sequence[int],
    name_suffix: str = "repaired",
    candidate_timeout_seconds: float | None = None,
    use_recovery_params: bool = False,
    model_parameter_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    source_run = Path(source_run)
    config = _load_json(source_run / "config.json")
    rounds = sorted(set(int(round_index) for round_index in retry_unavailable_rounds))
    if not rounds or rounds[0] < 1:
        raise ValueError("retry-unavailable-rounds must contain positive integers")
    transcript_path = source_run / "transcript.jsonl"
    if transcript_path.exists() or (
        source_run / "pending_batch_entries.jsonl"
    ).exists():
        unavailable_rounds = _unavailable_rounds(source_run)
        unmatched_rounds = sorted(set(rounds) - unavailable_rounds)
        if unmatched_rounds:
            raise ValueError(
                "requested repair rounds have no unavailable answers: "
                + ", ".join(str(value) for value in unmatched_rounds)
            )

    phases = config.get("protocol", {}).get("phases", [])
    if len(phases) != 1 or phases[0].get("kind") != "independent_judge_ranking":
        raise ValueError(
            "repair builder requires one independent_judge_ranking phase"
        )
    phase = phases[0]
    if not phase.get("probe_schedule"):
        raise ValueError("repair builder requires the adaptive probe schedule")

    phase["preauthored_probe_file"] = str(source_run / "transcript.jsonl")
    phase["preauthored_answer_file"] = str(source_run)
    phase["reuse_unavailable_answers"] = True
    phase["retry_unavailable_rounds"] = rounds
    phase["replay_source_targets"] = True
    phase.pop("preauthored_evidence_file", None)
    phase.pop("preauthored_ranking_file", None)
    providers = config.get("providers", {})
    if isinstance(providers, dict):
        candidate_provider = providers.get("openrouter_candidates")
        provider_rows = (
            [candidate_provider] if isinstance(candidate_provider, dict) else []
        )
    else:
        provider_rows = [
            provider
            for provider in providers
            if isinstance(provider, dict)
            and provider.get("name") == "openrouter_candidates"
        ]
    for provider in provider_rows:
        provider["request_retries"] = max(
            1, int(provider.get("request_retries", 0))
        )
        if candidate_timeout_seconds is not None:
            if candidate_timeout_seconds <= 0:
                raise ValueError("candidate timeout must be positive")
            provider["timeout_seconds"] = float(candidate_timeout_seconds)

    if use_recovery_params:
        participant_refs = {
            participant["model"] for participant in config.get("participants", [])
        }
        models = config.get("models", {})
        model_rows = models.values() if isinstance(models, dict) else models
        for model in model_rows:
            if model.get("name") not in participant_refs:
                continue
            recovery = model.get("recovery_params")
            if isinstance(recovery, dict) and recovery:
                model["params"] = deepcopy(recovery)

    applied_overrides = _apply_model_parameter_overrides(
        config,
        model_parameter_overrides or {},
    )
    config["name"] = f"{config['name']}_{name_suffix}"
    config.setdefault("metadata", {})["repair_source_run"] = str(source_run)
    config["metadata"]["repair_retry_unavailable_rounds"] = rounds
    config["metadata"]["repair_uses_recovery_params"] = use_recovery_params
    config["metadata"]["repair_parameter_overrides"] = applied_overrides
    return config


def _apply_model_parameter_overrides(
    config: dict[str, Any],
    overrides: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not overrides:
        return {}
    models = config.get("models", {})
    model_rows = list(models.values()) if isinstance(models, dict) else list(models)
    unmatched = set(overrides)
    applied: dict[str, dict[str, Any]] = {}
    for model in model_rows:
        keys = {str(model.get("name")), str(model.get("model"))}
        matched = next((key for key in keys if key in overrides), None)
        if matched is None:
            continue
        raw_params = overrides[matched]
        if not isinstance(raw_params, Mapping):
            raise ValueError(f"model parameter override for {matched} must be an object")
        params = deepcopy(dict(raw_params))
        model["params"] = params
        model["recovery_params"] = deepcopy(params)
        route = str(model.get("model") or model.get("name"))
        applied[route] = params
        unmatched.discard(matched)
    if unmatched:
        raise ValueError(
            "model parameter overrides did not match config models: "
            + ", ".join(sorted(unmatched))
        )
    return applied


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a replay config that retries unavailable answers."
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--retry-unavailable-rounds", required=True)
    parser.add_argument("--name-suffix", default="repaired")
    parser.add_argument("--candidate-timeout-seconds", type=float)
    parser.add_argument("--use-recovery-params", action="store_true")
    parser.add_argument(
        "--model-parameter-overrides-file",
        help=(
            "JSON object mapping a model name or provider route to replacement "
            "request parameters. Overrides also apply to visible-text recovery "
            "and are recorded in run metadata."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = build_replay_repair_config(
        args.source_run,
        retry_unavailable_rounds=[
            int(value)
            for value in args.retry_unavailable_rounds.split(",")
            if value.strip()
        ],
        name_suffix=args.name_suffix,
        candidate_timeout_seconds=args.candidate_timeout_seconds,
        use_recovery_params=args.use_recovery_params,
        model_parameter_overrides=(
            _load_json(Path(args.model_parameter_overrides_file))
            if args.model_parameter_overrides_file
            else None
        ),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "name": config["name"]}, indent=2))
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _unavailable_rounds(path: Path) -> set[int]:
    source_files = (
        [
            candidate
            for candidate in (
                path / "transcript.jsonl",
                path / "pending_batch_entries.jsonl",
            )
            if candidate.exists()
        ]
        if path.is_dir()
        else [path]
    )
    rounds = set()
    for source_file in source_files:
        with source_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("metadata", {}).get("answer_unavailable"):
                    continue
                round_index = row.get("round_index")
                if isinstance(round_index, int) and round_index > 0:
                    rounds.add(round_index)
    return rounds


if __name__ == "__main__":
    raise SystemExit(main())
