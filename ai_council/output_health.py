from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping


@dataclass(frozen=True)
class OutputHealth:
    visible_chars: int
    completion_tokens: int | None
    reasoning_tokens: int | None

    @property
    def has_visible_text(self) -> bool:
        return self.visible_chars > 0

    @property
    def reasoning_token_ratio(self) -> float | None:
        if self.completion_tokens is None or self.completion_tokens <= 0:
            return None
        if self.reasoning_tokens is None:
            return None
        return self.reasoning_tokens / self.completion_tokens

    @property
    def reasoning_dominated(self) -> bool:
        ratio = self.reasoning_token_ratio
        return ratio is not None and ratio >= 0.75


def inspect_output_health(content: str, usage: Mapping[str, Any] | None) -> OutputHealth:
    usage = usage or {}
    return OutputHealth(
        visible_chars=len(content.strip()),
        completion_tokens=_optional_int(usage.get("completion_tokens")),
        reasoning_tokens=_extract_reasoning_tokens(usage),
    )


def format_output_health_evidence(health: OutputHealth) -> str:
    ratio = health.reasoning_token_ratio
    ratio_text = f"{ratio:.2f}" if ratio is not None else "unknown"
    return (
        f"visible_chars={health.visible_chars}; "
        f"completion_tokens={health.completion_tokens}; "
        f"reasoning_tokens={health.reasoning_tokens}; "
        f"reasoning_token_ratio={ratio_text}"
    )


def _extract_reasoning_tokens(usage: Mapping[str, Any]) -> int | None:
    details = usage.get("completion_tokens_details")
    if isinstance(details, Mapping):
        value = details.get("reasoning_tokens")
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
    return _optional_int(usage.get("reasoning_tokens"))


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Real):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
