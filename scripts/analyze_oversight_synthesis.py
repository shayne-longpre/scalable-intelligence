from __future__ import annotations

import argparse
import json

from ai_council.oversight_synthesis import build_frontier_synthesis


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pool two or more oversight-frontier study waves."
    )
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--published-json")
    args = parser.parse_args()

    summary = build_frontier_synthesis(
        result_paths=args.result,
        output_dir=args.output_dir,
        published_json_path=args.published_json,
    )
    print(json.dumps(summary["pooled"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
