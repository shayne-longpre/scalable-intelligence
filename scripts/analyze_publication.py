from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_council.publication_analysis import (
    build_publication_analysis,
    write_publication_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROSSED_REPORT = (
    ROOT
    / "runs"
    / "report_cards"
    / "catalog_ladder50_crossed_20260721_v1"
    / "report_card_summary.json"
)
DEFAULT_CATALOG_ORDER_ANALYSIS = (
    ROOT
    / "runs"
    / "20260725T203344Z_catalog_ladder50_order_audit_fable_on_sol_seed_20260814"
    / "analysis_summary.json"
)
DEFAULT_OVERSIGHT_SYNTHESIS = (
    ROOT / "data" / "oversight_frontier_synthesis_matched_results.json"
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build publication robustness and probe-comparison outputs."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "data" / "publication_analysis.json",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "docs" / "figures" / "publication",
    )
    parser.add_argument(
        "--robustness-markdown",
        type=Path,
        default=ROOT / "docs" / "robustness_appendix.md",
    )
    args = parser.parse_args()

    catalog = load_json(ROOT / "data" / "model_catalog.openrouter.json")
    summary = build_publication_analysis(
        self_study=load_json(ROOT / "data" / "probe_self_study_results.json"),
        probe_catalog=load_json(ROOT / "data" / "probe_catalog.json"),
        taxonomy=load_json(ROOT / "data" / "evaluation_taxonomy.json"),
        probe_scores=load_json(
            ROOT / "data" / "catalog_ladder50_probe_scores.json"
        ),
        crossed_report=load_json(DEFAULT_CROSSED_REPORT),
        research_summary=load_json(
            ROOT / "data" / "research_question_synthesis.json"
        ),
        catalog_stability=load_json(
            ROOT / "data" / "catalog_ladder50_opening10_stability.json"
        ),
        oversight_synthesis=load_json(DEFAULT_OVERSIGHT_SYNTHESIS),
        oversight_order_replay=load_json(
            ROOT / "data" / "oversight_frontier_v1_order_replay_results.json"
        ),
        verifier_council=load_json(
            ROOT / "data" / "verifier_council_matched_v1_results.json"
        ),
        catalog_order_analysis=load_json(DEFAULT_CATALOG_ORDER_ANALYSIS),
    )
    display_names = {
        row["provider_model_id"]: row["display_name"]
        for row in catalog["models"]
    }
    write_publication_outputs(
        summary=summary,
        output_json=args.output_json,
        figure_dir=args.figure_dir,
        robustness_markdown=args.robustness_markdown,
        display_names=display_names,
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "figure_dir": str(args.figure_dir),
                "robustness_markdown": str(args.robustness_markdown),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
