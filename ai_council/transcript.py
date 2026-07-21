from __future__ import annotations

from ai_council.core import TranscriptEntry


class Transcript:
    def __init__(self) -> None:
        self.entries: list[TranscriptEntry] = []

    def append(self, entry: TranscriptEntry) -> None:
        self.entries.append(entry)

    def public_entries(self) -> list[TranscriptEntry]:
        return [entry for entry in self.entries if entry.visibility == "public"]

    def entries_for_phase(self, phase: str) -> list[TranscriptEntry]:
        return [entry for entry in self.entries if entry.phase == phase]

    def entries_for_stream(self, stream_id: str) -> list[TranscriptEntry]:
        return [entry for entry in self.entries if entry.metadata.get("stream_id") == stream_id]

    def private_entries_for(self, participant_id: str) -> list[TranscriptEntry]:
        return [
            entry
            for entry in self.entries
            if entry.visibility == "private" and entry.speaker == participant_id
        ]

    def render_public(self, max_turns: int = 80) -> str:
        return render_entries(self.public_entries()[-max_turns:])

    def render_private(self, participant_id: str, max_turns: int = 20) -> str:
        return render_entries(self.private_entries_for(participant_id)[-max_turns:])


def render_entries(entries: list[TranscriptEntry]) -> str:
    if not entries:
        return "(none yet)"
    lines = []
    for entry in entries:
        round_part = f", round {entry.round_index}" if entry.round_index is not None else ""
        lines.append(
            f"Turn {entry.turn_id} [{entry.phase}{round_part}] {entry.speaker}: {entry.content}"
        )
    return "\n\n".join(lines)
