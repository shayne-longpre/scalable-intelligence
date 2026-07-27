from __future__ import annotations

import argparse
import json

from ai_council.probe_catalog import build_probe_catalog, write_probe_catalog


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deduplicated, provenance-preserving catalog of accepted "
            "judge-authored probes."
        )
    )
    parser.add_argument("--study-summary", required=True)
    parser.add_argument("--model-catalog", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    catalog = build_probe_catalog(
        args.study_summary,
        model_catalog=args.model_catalog,
    )
    write_probe_catalog(catalog, args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "accepted_runs": catalog["accepted_run_count"],
                "probe_occurrences": catalog["probe_occurrence_count"],
                "unique_probes": catalog["unique_probe_count"],
                "author_models": catalog["author_model_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
