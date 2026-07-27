from __future__ import annotations

import argparse
import json

from ai_council.research_synthesis import build_research_synthesis


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build publication-level summaries for catalog ranking and the "
            "oversight frontier from archived study outputs."
        )
    )
    parser.add_argument(
        "--catalog-stability",
        default="data/catalog_ladder50_opening5_stability.json",
    )
    parser.add_argument(
        "--oversight-synthesis",
        default="data/oversight_frontier_synthesis_matched_results.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--published-json")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    summary = build_research_synthesis(
        catalog_stability_path=args.catalog_stability,
        oversight_synthesis_path=args.oversight_synthesis,
        output_dir=args.output_dir,
        published_json_path=args.published_json,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "catalog_mean_pairwise_accuracy": summary[
                    "rq1_catalog_ranking"
                ]["mean_pairwise_accuracy"],
                "superior_recognition_rate": summary[
                    "rq2_oversight_frontier"
                ]["superior_recognition_rate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
