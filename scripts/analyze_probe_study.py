from __future__ import annotations

import argparse
import json

from ai_council.probe_study import build_probe_study


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze how independent judges design and adapt probes."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run specification as COHORT=RUN_DIR.",
    )
    parser.add_argument("--catalog", default="data/model_catalog.openrouter.json")
    parser.add_argument("--taxonomy", default="data/evaluation_taxonomy.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--published-json")
    args = parser.parse_args()
    summary = build_probe_study(
        run_specs=[_parse_run_spec(value) for value in args.run],
        catalog_path=args.catalog,
        output_dir=args.output_dir,
        published_json_path=args.published_json,
        taxonomy_path=args.taxonomy,
    )
    print(
        json.dumps(
            {
                "run_count": summary["run_count"],
                "probe_count": summary["probe_count"],
                "cohorts": list(summary["cohorts"]),
                "output_dir": args.output_dir,
            },
            indent=2,
        )
    )
    return 0


def _parse_run_spec(value: str) -> dict[str, str]:
    cohort, separator, run_dir = value.partition("=")
    if not separator or not cohort.strip() or not run_dir.strip():
        raise ValueError("--run must use COHORT=RUN_DIR")
    return {"cohort": cohort.strip(), "run_dir": run_dir.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
