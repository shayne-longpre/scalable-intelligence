from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_council.config import ExperimentConfig
from ai_council.core import TranscriptEntry
from ai_council.prompts import PROMPT_SET_VERSION, PromptLibrary


class RunStore:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.transcript_path = run_dir / "transcript.jsonl"
        self.findings_path = run_dir / "monitor_findings.jsonl"

    @classmethod
    def create(cls, base_dir: str | Path, config: ExperimentConfig) -> "RunStore":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in config.name)
        run_dir = _unique_run_dir(Path(base_dir), f"{timestamp}_{safe_name}")
        run_dir.mkdir(parents=True, exist_ok=False)
        store = cls(run_dir)
        store.write_json("config.json", config)
        prompts = PromptLibrary(config.prompt_overrides).snapshot()
        canonical = json.dumps(
            prompts,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        store.write_json(
            "prompt_snapshot.json",
            {
                "version": PROMPT_SET_VERSION,
                "sha256": hashlib.sha256(canonical).hexdigest(),
                "prompts": prompts,
            },
        )
        return store

    def append_entry(self, entry: TranscriptEntry) -> None:
        with self.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def append_finding(self, finding: Any) -> None:
        with self.findings_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_jsonable(finding), ensure_ascii=False) + "\n")

    def write_json(self, name: str, data: Any) -> None:
        with (self.run_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(data), handle, indent=2, ensure_ascii=False)
            handle.write("\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _to_jsonable(data: Any) -> Any:
    if is_dataclass(data):
        return asdict(data)
    if isinstance(data, dict):
        return {key: _to_jsonable(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_to_jsonable(item) for item in data]
    if isinstance(data, tuple):
        return [_to_jsonable(item) for item in data]
    return data


def _unique_run_dir(base_dir: Path, stem: str) -> Path:
    candidate = base_dir / stem
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = base_dir / f"{stem}_{suffix}"
    return candidate
