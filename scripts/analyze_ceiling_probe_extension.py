from __future__ import annotations

import argparse
import json

from ai_council.ceiling_probe_analysis import build_ceiling_probe_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare archived opening rankings with ceiling-aware probe "
            "extensions."
        )
    )
    parser.add_argument("--study", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument(
        "--catalog", default="data/model_catalog.openrouter.json"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--published-json")
    args = parser.parse_args()
    summary = build_ceiling_probe_report(
        study_path=args.study,
        runs_root=args.runs_root,
        catalog_path=args.catalog,
        output_dir=args.output_dir,
        published_json_path=args.published_json,
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "condition_count": summary["condition_count"],
                "new_probe_count": summary["new_probe_count"],
                "mean_accuracy_delta_vs_opening": summary[
                    "mean_accuracy_delta_vs_opening"
                ],
                "reported_cost_usd": summary["reported_cost_usd"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
