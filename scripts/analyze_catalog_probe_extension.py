from __future__ import annotations

import argparse
import json

from ai_council.catalog_probe_extension import (
    build_catalog_probe_extension_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare five-probe catalog runs with ten-probe extensions."
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("LABEL", "SOURCE_RUN", "EXTENSION_RUN"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--published-json")
    args = parser.parse_args()
    summary = build_catalog_probe_extension_report(
        pairs=args.pair,
        output_dir=args.output_dir,
        published_json_path=args.published_json,
    )
    print(
        json.dumps(
            {
                "conditions": len(summary["conditions"]),
                "output_dir": args.output_dir,
                **summary["aggregate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
