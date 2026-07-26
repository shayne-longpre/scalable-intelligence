from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_council.experiment_builders import (
    build_exact_evidence_order_replay_config,
)


def build_order_replay_study_configs(
    study_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    study = _load_json(study_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for condition in study["conditions"]:
        source_run = Path(condition["source_run"])
        if not (source_run / "config.json").exists():
            raise ValueError(
                f"{condition['id']} source run has no config: {source_run}"
            )
        if not (source_run / "transcript.jsonl").exists():
            raise ValueError(
                f"{condition['id']} source run has no transcript: {source_run}"
            )
        config = build_exact_evidence_order_replay_config(
            _load_json(source_run / "config.json"),
            source_run=source_run,
            comparison_seed=int(condition["comparison_seed"]),
            study_condition=condition["id"],
        )
        config["metadata"]["study_file"] = str(study_path)
        config["metadata"]["research_question"] = study["research_question"]
        config_path = output_dir / f"{condition['id']}.json"
        config_path.write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )
        generated.append(
            {
                "condition_id": condition["id"],
                "source_run": str(source_run),
                "config": str(config_path),
                "comparison_seed": condition["comparison_seed"],
            }
        )

    index = {
        "study": study["name"],
        "study_file": str(study_path),
        "configs": generated,
    }
    (output_dir / "index.json").write_text(
        json.dumps(index, indent=2) + "\n",
        encoding="utf-8",
    )
    return index


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build exact-evidence answer-order replay configs."
    )
    parser.add_argument("--study", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    index = build_order_replay_study_configs(
        Path(args.study),
        Path(args.output_dir),
    )
    print(json.dumps(index, indent=2))
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
