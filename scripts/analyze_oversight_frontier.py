from __future__ import annotations

import argparse
import json

from ai_council.oversight_analysis import build_oversight_study_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze completed conditions from an oversight-frontier study."
    )
    parser.add_argument("--study", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--catalog")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--published-json")
    parser.add_argument("--probe-audit")
    args = parser.parse_args()

    summary = build_oversight_study_report(
        study_path=args.study,
        runs_root=args.runs_root,
        output_dir=args.output_dir,
        catalog_path=args.catalog,
        published_json_path=args.published_json,
        probe_audit_path=args.probe_audit,
    )
    print(
        json.dumps(
            {
                "study": summary["study"],
                "condition_count": len(summary["conditions"]),
                "aggregate": summary["aggregate"],
                "output_dir": args.output_dir,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
