from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from ai_council.probe_study import build_probe_study


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze how independent judges design and adapt probes."
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run specification as COHORT=RUN_DIR.",
    )
    parser.add_argument(
        "--extension-run",
        action="append",
        default=[],
        help=(
            "Run specification as COHORT=RUN_DIR, excluding replayed opening "
            "probes using the run's archived_opening_probe_count metadata."
        ),
    )
    parser.add_argument(
        "--base-summary",
        help="Reuse the accepted runs in an existing probe-study summary.",
    )
    parser.add_argument("--catalog", default="data/model_catalog.openrouter.json")
    parser.add_argument("--taxonomy", default="data/evaluation_taxonomy.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--published-json")
    args = parser.parse_args()
    run_specs = collect_run_specs(
        args.base_summary,
        args.run,
        extension_values=args.extension_run,
    )
    summary = build_probe_study(
        run_specs=run_specs,
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


def collect_run_specs(
    base_summary: str | None,
    run_values: list[str],
    *,
    extension_values: list[str] | None = None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if base_summary:
        summary = json.loads(Path(base_summary).read_text(encoding="utf-8"))
        rows = summary.get("runs")
        if not isinstance(rows, list):
            raise ValueError("base summary must contain a runs list")
        specs.extend(_run_spec_from_summary(row) for row in rows)
    specs.extend(_parse_run_spec(value) for value in run_values)
    specs.extend(
        _extension_run_spec(value) for value in (extension_values or [])
    )
    if not specs:
        raise ValueError("provide --base-summary, --run, or both")
    seen: dict[Path, str] = {}
    for spec in specs:
        resolved = Path(spec["run_dir"]).resolve()
        prior = seen.get(resolved)
        if prior is not None:
            raise ValueError(
                f"duplicate run directory {spec['run_dir']} "
                f"(cohorts {prior!r} and {spec['cohort']!r})"
            )
        seen[resolved] = spec["cohort"]
    return specs


def _parse_run_spec(value: str) -> dict[str, str]:
    cohort, separator, run_dir = value.partition("=")
    if not separator or not cohort.strip() or not run_dir.strip():
        raise ValueError("--run must use COHORT=RUN_DIR")
    return {"cohort": cohort.strip(), "run_dir": run_dir.strip()}


def _run_spec_from_summary(row: Any) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("base summary run entries must be objects")
    cohort = row.get("cohort")
    run_dir = row.get("run_dir")
    if not isinstance(cohort, str) or not cohort.strip():
        raise ValueError("base summary run cohort must be a non-empty string")
    if not isinstance(run_dir, str) or not run_dir.strip():
        raise ValueError("base summary run_dir must be a non-empty string")
    spec: dict[str, Any] = {
        "cohort": cohort.strip(),
        "run_dir": run_dir.strip(),
    }
    sequence_min = row.get("probe_sequence_min")
    if sequence_min is not None:
        spec["probe_sequence_min"] = int(sequence_min)
    return spec


def _extension_run_spec(value: str) -> dict[str, Any]:
    spec: dict[str, Any] = _parse_run_spec(value)
    config_path = Path(spec["run_dir"]) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    opening_count = config.get("metadata", {}).get(
        "archived_opening_probe_count"
    )
    if not isinstance(opening_count, int) or opening_count < 1:
        raise ValueError(
            f"extension run {spec['run_dir']} has no valid "
            "archived_opening_probe_count"
        )
    spec["probe_sequence_min"] = opening_count + 1
    return spec


if __name__ == "__main__":
    raise SystemExit(main())
