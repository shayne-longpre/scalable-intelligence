from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Visibility = Literal["public", "private", "monitor"]


@dataclass(frozen=True)
class ModelRequest:
    model: str
    messages: list[dict[str, str]]
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    content: str
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    provider: str | None = None


@dataclass
class TranscriptEntry:
    turn_id: int
    phase: str
    speaker: str
    visibility: Visibility
    content: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    round_index: int | None = None
    parsed: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptEntry":
        return cls(**data)


@dataclass(frozen=True)
class MonitorFinding:
    code: str
    severity: str
    message: str
    speaker: str
    turn_id: int | None = None
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
