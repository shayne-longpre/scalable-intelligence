from __future__ import annotations

import argparse
import json

from ai_council.catalog_stability import build_catalog_stability_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two matched catalog-ladder report summaries."
    )
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--replication-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--published-json")
    args = parser.parse_args()
    summary = build_catalog_stability_report(
        baseline_summary_path=args.baseline_summary,
        replication_summary_path=args.replication_summary,
        output_dir=args.output_dir,
        published_json_path=args.published_json,
    )
    print(
        json.dumps(
            {
                "judges": len(summary["judges"]),
                "mean_rank_replication_tau": summary[
                    "mean_rank_replication_tau"
                ],
                "output_dir": args.output_dir,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
