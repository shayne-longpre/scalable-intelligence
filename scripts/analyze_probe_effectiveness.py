from __future__ import annotations

import argparse
import json

from ai_council.probe_effectiveness import build_probe_effectiveness_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether probe taxonomy labels predict diagnosticity for "
            "held-out probe authors."
        )
    )
    parser.add_argument(
        "--self-study",
        default="data/probe_self_study_results.json",
    )
    parser.add_argument(
        "--probe-catalog",
        default="data/probe_catalog.json",
    )
    parser.add_argument(
        "--probe-evolution",
        default="data/probe_evolution_results.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--published-json")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    summary = build_probe_effectiveness_report(
        self_study_path=args.self_study,
        probe_catalog_path=args.probe_catalog,
        probe_evolution_path=args.probe_evolution,
        output_dir=args.output_dir,
        published_json_path=args.published_json,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "probe_count": summary["probe_count"],
                "mean_pair_accuracy": summary[
                    "mean_candidate_pair_accuracy"
                ],
                "held_out": {
                    row["name"]: row["mean_author_concordance"]
                    for row in summary["held_out_label_prediction"]
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
