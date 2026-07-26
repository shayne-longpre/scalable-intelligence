from __future__ import annotations

import argparse
import json

from ai_council.oversight_replication import build_replication_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine two oversight-frontier studies and their order audit."
    )
    parser.add_argument("--first", required=True)
    parser.add_argument("--replication", required=True)
    parser.add_argument("--order-replay", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--published-json")
    args = parser.parse_args()

    summary = build_replication_report(
        first_result_path=args.first,
        replication_result_path=args.replication,
        order_replay_path=args.order_replay,
        output_dir=args.output_dir,
        published_json_path=args.published_json,
    )
    print(json.dumps(summary["pooled"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
