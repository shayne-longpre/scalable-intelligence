from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
from threading import Lock
from time import perf_counter

from ai_council.analysis import analyze_run
from ai_council.clients import build_clients
from ai_council.config import ConfigError, load_experiment_config
from ai_council.core import ModelRequest
from ai_council.env import load_dotenv
from ai_council.orchestrator import CouncilRunner
from ai_council.report_card import build_report_card
from ai_council.storage import RunStore
from ai_council.validation import revalidate_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai-council")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config")
    validate_parser.add_argument("--config", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--output-dir")

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--config", required=True)
    smoke_parser.add_argument("--output")

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--run-dir", required=True)
    analyze_parser.add_argument("--prior-ranking-file")

    revalidate_parser = subparsers.add_parser("revalidate")
    revalidate_parser.add_argument("--run-dir", required=True)

    report_parser = subparsers.add_parser("report-card")
    report_parser.add_argument("--run-dir", action="append", required=True)
    report_parser.add_argument("--prior-ranking-file")
    report_parser.add_argument("--output-dir")
    report_parser.add_argument("--llm-summary-config")

    args = parser.parse_args(argv)
    load_dotenv()

    try:
        if args.command == "validate-config":
            config = load_experiment_config(args.config)
            print(json.dumps({"ok": True, "name": config.name}, indent=2))
            return 0

        if args.command == "run":
            config = load_experiment_config(args.config)
            output_dir = Path(args.output_dir or config.run.output_dir)
            clients = build_clients(config)
            store = RunStore.create(output_dir, config)
            CouncilRunner(config, clients, store).run()
            print(json.dumps({"run_dir": str(store.run_dir)}, indent=2))
            return 0

        if args.command == "smoke":
            config = load_experiment_config(args.config)
            clients = build_clients(config)
            seen_model_names = sorted(
                {agent.model for agent in [*config.participants, *config.judges]}
            )
            provider_locks = {name: Lock() for name in clients}

            def smoke_model(model_name: str) -> dict[str, object]:
                model = config.models[model_name]
                client = clients[model.provider]
                guard = (
                    nullcontext()
                    if client.supports_parallel_requests
                    else provider_locks[model.provider]
                )
                started = perf_counter()
                try:
                    with guard:
                        response = client.generate(
                            ModelRequest(
                                model=model.model,
                                messages=[
                                    {"role": "user", "content": "Reply with exactly: smoke ok"}
                                ],
                                params={
                                    **model.params,
                                    "max_tokens": _smoke_max_tokens(model.params),
                                },
                                metadata={"smoke": True},
                            )
                        )
                except Exception as exc:
                    return {
                        "model_ref": model_name,
                        "provider": model.provider,
                        "model": model.model,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_seconds": perf_counter() - started,
                    }
                return {
                    "model_ref": model_name,
                    "provider": model.provider,
                    "model": response.model,
                    "ok": bool(response.content.strip()),
                    "instruction_ok": "smoke ok" in response.content.lower(),
                    "content": response.content.strip(),
                    "usage": response.usage,
                    "latency_seconds": perf_counter() - started,
                }

            with ThreadPoolExecutor(max_workers=config.run.max_parallel_calls) as executor:
                results = list(executor.map(smoke_model, seen_model_names))
            payload = {"ok": all(result["ok"] for result in results), "results": results}
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(payload, indent=2))
            return 0

        if args.command == "analyze":
            summary = analyze_run(args.run_dir, prior_ranking_file=args.prior_ranking_file)
            print(json.dumps(summary, indent=2))
            return 0

        if args.command == "revalidate":
            summary = revalidate_run(args.run_dir)
            print(json.dumps(summary, indent=2))
            return 0

        if args.command == "report-card":
            summary = build_report_card(
                args.run_dir,
                prior_ranking_file=args.prior_ranking_file,
                output_dir=args.output_dir,
                llm_summary_config=args.llm_summary_config,
            )
            print(json.dumps(summary, indent=2))
            return 0
    except ConfigError as exc:
        parser.error(str(exc))
    return 1


def _smoke_max_tokens(params: dict[str, object]) -> int:
    reasoning = params.get("reasoning")
    if not isinstance(reasoning, dict):
        return 512
    explicit_budget = _positive_int(reasoning.get("max_tokens")) or 0
    effort = str(reasoning.get("effort", "")).lower()
    effort_floor = 2048 if effort in {"high", "xhigh", "max"} else 1024
    return max(512, explicit_budget + 128, effort_floor)


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
