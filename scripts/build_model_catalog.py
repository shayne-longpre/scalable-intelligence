from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_council.model_catalog import (
    ARTIFICIAL_ANALYSIS_MODELS_URL,
    OPENROUTER_MODELS_URL,
    build_openrouter_catalog,
    parse_artificial_analysis_models_html,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the OpenRouter model intelligence catalog.")
    parser.add_argument("--openrouter-models-json", help="Cached OpenRouter /models JSON.")
    parser.add_argument("--artificial-analysis-html", help="Cached Artificial Analysis leaderboard HTML.")
    parser.add_argument("--output", default="data/model_catalog.openrouter.json")
    args = parser.parse_args()

    openrouter_payload = _load_json(args.openrouter_models_json) if args.openrouter_models_json else _fetch_openrouter()
    artificial_analysis_html = (
        Path(args.artificial_analysis_html).read_text(encoding="utf-8")
        if args.artificial_analysis_html
        else _fetch_text(ARTIFICIAL_ANALYSIS_MODELS_URL)
    )
    artificial_analysis_models = parse_artificial_analysis_models_html(artificial_analysis_html)
    catalog = build_openrouter_catalog(openrouter_payload, artificial_analysis_models)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": catalog["summary"]}, indent=2))
    return 0


def _load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fetch_openrouter() -> dict:
    headers = {}
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(OPENROUTER_MODELS_URL, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "ai-council-catalog-builder/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
