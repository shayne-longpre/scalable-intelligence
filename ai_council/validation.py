from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_council.config import ExperimentConfig
from ai_council.core import MonitorFinding, TranscriptEntry
from ai_council.json_tools import JsonExtractionError, extract_json_object
from ai_council.monitors import RuleBasedMonitor
from ai_council.orchestrator import (
    INDEPENDENT_JUDGE_COMPARISON_KEYS,
    INDEPENDENT_JUDGE_EVIDENCE_KEYS,
    INDEPENDENT_JUDGE_RANKING_KEYS,
    INDEPENDENT_JUDGE_WAVE_KEYS,
    ROUND_ROBIN_ASSESSMENT_KEYS,
    ROUND_ROBIN_MEMORY_KEYS,
    ROUND_ROBIN_RANKING_KEYS,
)
from ai_council.storage import load_jsonl


def revalidate_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    with (run_path / "config.json").open("r", encoding="utf-8") as handle:
        config = ExperimentConfig.from_dict(json.load(handle))

    monitor = RuleBasedMonitor(
        identity_terms=config.monitor.identity_terms,
        strict=config.monitor.strict,
    )
    participant_ids = [participant.id for participant in config.participants]
    phases = {phase.name: phase for phase in config.protocol.phases}
    findings: list[MonitorFinding] = []

    for data in load_jsonl(run_path / "transcript.jsonl"):
        entry = TranscriptEntry.from_dict(data)
        phase = phases.get(entry.phase)
        parsed = entry.parsed
        require_json = _entry_requires_json(entry, phase)
        required_keys = _entry_required_keys(entry, phase)
        if phase and require_json and parsed is None:
            try:
                parsed = extract_json_object(entry.content)
            except JsonExtractionError:
                parsed = None

        findings.extend(monitor.check_entry(entry))
        if phase:
            findings.extend(
                monitor.check_required_keys(
                    entry,
                    parsed,
                    required_keys,
                    require_json=require_json,
                )
            )
            if require_json:
                findings.extend(monitor.check_structured_values(entry, parsed, participant_ids))

    findings_path = run_path / "revalidation_findings.jsonl"
    with findings_path.open("w", encoding="utf-8") as handle:
        for finding in findings:
            handle.write(json.dumps(finding.to_dict(), ensure_ascii=False) + "\n")

    summary = {
        "finding_count": len(findings),
        "findings_path": str(findings_path),
        "codes": _counts(finding.code for finding in findings),
    }
    with (run_path / "revalidation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _entry_requires_json(entry: TranscriptEntry, phase) -> bool:
    if phase is None:
        return False
    if phase.kind == "separate_interviews":
        return phase.require_json and entry.metadata.get("interaction_role") == "assessment"
    if phase.kind == "round_robin_probes":
        return entry.metadata.get("interaction_role") in {
            "assessment",
            "round_ranking",
            "memory_update",
        }
    if phase.kind == "independent_judge_ranking":
        return entry.metadata.get("interaction_role") in {
            "evidence_card",
            "judge_ranking",
            "probe_comparison",
            "wave_judgment",
        }
    return phase.require_json


def _entry_required_keys(entry: TranscriptEntry, phase) -> list[str]:
    if phase is None:
        return []
    if phase.kind == "separate_interviews" and entry.metadata.get("interaction_role") != "assessment":
        return []
    if phase.kind == "round_robin_probes":
        role = entry.metadata.get("interaction_role")
        if role == "assessment":
            return ROUND_ROBIN_ASSESSMENT_KEYS
        if role == "round_ranking":
            return ROUND_ROBIN_RANKING_KEYS
        if role == "memory_update":
            return ROUND_ROBIN_MEMORY_KEYS
        return []
    if phase.kind == "independent_judge_ranking":
        return {
            "evidence_card": INDEPENDENT_JUDGE_EVIDENCE_KEYS,
            "judge_ranking": INDEPENDENT_JUDGE_RANKING_KEYS,
            "probe_comparison": INDEPENDENT_JUDGE_COMPARISON_KEYS,
            "wave_judgment": INDEPENDENT_JUDGE_WAVE_KEYS,
        }.get(entry.metadata.get("interaction_role"), [])
    return phase.required_keys
