from __future__ import annotations

import json
from typing import Any

from ai_council.config import ContextSpec
from ai_council.core import TranscriptEntry
from ai_council.transcript import Transcript, render_entries


def render_context_sections(
    transcript: Transcript,
    participant_id: str,
    context: ContextSpec,
    *,
    default_public_turns: int,
    stream_id: str | None = None,
    private_scope: str = "default",
) -> tuple[str, str]:
    public_limit = default_public_turns if context.max_public_turns is None else context.max_public_turns
    private_limit = context.max_private_turns
    stream_context = _render_stream_context(transcript, stream_id, participant_id, context.max_stream_turns)
    public_context = render_entries(transcript.public_entries()[-public_limit:]) if public_limit else "(omitted)"

    if private_scope == "stream_only":
        return (public_context, stream_context or "(omitted)")
    if private_scope != "default":
        raise ValueError(f"unknown private scope: {private_scope}")

    if context.mode == "transcript":
        private_context = transcript.render_private(participant_id, private_limit)
        return (public_context, _join_context(private_context, stream_context))
    if context.mode == "private_memory":
        private_memory = render_private_memory(transcript.private_entries_for(participant_id)[-private_limit:])
        return (public_context, _join_context(private_memory, stream_context))
    raise ValueError(f"unknown context mode: {context.mode}")


def render_private_memory(entries: list[TranscriptEntry]) -> str:
    if not entries:
        return "(none yet)"
    return "\n\n".join(_memory_entry(entry) for entry in entries)


def _memory_entry(entry: TranscriptEntry) -> str:
    if isinstance(entry.parsed, dict):
        body = _compact_json_memory(entry.parsed)
        return f"Private memory turn {entry.turn_id} [{entry.phase}] {entry.speaker}:\n{body}"
    return (
        f"Private note turn {entry.turn_id} [{entry.phase}] {entry.speaker}:\n"
        f"{entry.content.strip()}"
    )


def _compact_json_memory(parsed: dict[str, Any]) -> str:
    keys = [
        "participant_id",
        "phase",
        "round_index",
        "interviewer_id",
        "respondent_id",
        "target_participant_id",
        "question_summary",
        "answer_summary",
        "assessment",
        "qa_assessment_summaries",
        "current_ranking",
        "ranking",
        "ranking_summary",
        "confidence",
        "criteria",
        "evidence",
        "uncertainties",
        "updates",
        "next_probe",
        "next_probe_strategy",
        "next_evidence_needed",
        "next_round_plan",
    ]
    compact = {key: parsed[key] for key in keys if key in parsed}
    if not compact:
        compact = parsed
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _render_stream_context(
    transcript: Transcript,
    stream_id: str | None,
    participant_id: str,
    max_turns: int,
) -> str:
    if not stream_id:
        return ""
    if max_turns <= 0:
        return "Stream-local history:\n(omitted)"
    entries = [
        entry
        for entry in transcript.entries_for_stream(stream_id)
        if _stream_entry_visible_to(entry, participant_id)
    ][-max_turns:]
    return "Stream-local history:\n" + render_entries(entries)


def _join_context(private_context: str, stream_context: str) -> str:
    if not stream_context:
        return private_context
    return f"{private_context}\n\n{stream_context}"


def _stream_entry_visible_to(entry: TranscriptEntry, participant_id: str) -> bool:
    if entry.speaker == participant_id:
        return True
    return entry.metadata.get("interaction_role") in {"question", "answer", "discussion"}
