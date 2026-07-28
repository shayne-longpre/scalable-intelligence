from __future__ import annotations

import argparse
import json

from ai_council.judge_council import build_judge_council_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze matched verifier-oriented batteries and independent "
            "judge councils."
        )
    )
    parser.add_argument("--study", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--published-json")
    args = parser.parse_args()
    summary = build_judge_council_report(
        study_path=args.study,
        output_dir=args.output_dir,
        published_json_path=args.published_json,
    )
    print(
        json.dumps(
            {
                "study": summary["study"],
                "conditions": summary["conditions"],
                "matched_effects": summary["matched_effects"],
                "reported_evaluator_cost_usd": summary[
                    "reported_evaluator_cost_usd"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
