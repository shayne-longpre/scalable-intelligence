from __future__ import annotations

import argparse
import json

from ai_council.order_replay_analysis import build_order_replay_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze exact-evidence answer-order replay runs."
    )
    parser.add_argument("--study", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--published-json")
    args = parser.parse_args()

    summary = build_order_replay_report(
        study_path=args.study,
        runs_root=args.runs_root,
        output_dir=args.output_dir,
        published_json_path=args.published_json,
    )
    print(
        json.dumps(
            {
                "study": summary["study"],
                "aggregate": summary["aggregate"],
                "output_dir": args.output_dir,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
