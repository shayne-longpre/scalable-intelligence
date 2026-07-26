from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def build_replay_repair_config(
    source_run: str | Path,
    *,
    retry_unavailable_rounds: Sequence[int],
    name_suffix: str = "repaired",
) -> dict[str, Any]:
    source_run = Path(source_run)
    config = _load_json(source_run / "config.json")
    rounds = sorted(set(int(round_index) for round_index in retry_unavailable_rounds))
    if not rounds or rounds[0] < 1:
        raise ValueError("retry-unavailable-rounds must contain positive integers")

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

    config["name"] = f"{config['name']}_{name_suffix}"
    config.setdefault("metadata", {})["repair_source_run"] = str(source_run)
    config["metadata"]["repair_retry_unavailable_rounds"] = rounds
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a replay config that retries unavailable answers."
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--retry-unavailable-rounds", required=True)
    parser.add_argument("--name-suffix", default="repaired")
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
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "name": config["name"]}, indent=2))
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
