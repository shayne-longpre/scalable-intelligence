from __future__ import annotations

import re
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from ai_council.core import MonitorFinding, TranscriptEntry
from ai_council.output_health import format_output_health_evidence, inspect_output_health
from ai_council.rankings import duplicate_rank_positions, is_supported_ranking_shape, ranking_ids


DEFAULT_IDENTITY_TERMS = [
    "gpt-",
    "gpt ",
    "chatgpt",
    "openai",
    "claude",
    "anthropic",
    "gemini",
    "google deepmind",
    "llama",
    "mistral",
    "deepseek",
    "qwen",
    "my training data",
    "my training cutoff",
    "my knowledge cutoff",
    "I was trained by",
    "my benchmark score",
]

LIST_OF_STRINGS_FIELDS = {
    "comparative_evidence",
    "criteria",
    "evidence",
    "follow_up_rationale",
    "uncertainties",
    "updates",
    "next_probe_strategy",
    "next_evidence_needed",
    "strengths",
    "weaknesses",
}


@dataclass
class RuleBasedMonitor:
    identity_terms: list[str] = field(default_factory=list)
    strict: bool = False

    def __post_init__(self) -> None:
        terms = self.identity_terms or DEFAULT_IDENTITY_TERMS
        self._identity_pattern = re.compile("|".join(re.escape(term) for term in terms), re.IGNORECASE)

    def check_entry(self, entry: TranscriptEntry) -> list[MonitorFinding]:
        findings: list[MonitorFinding] = []
        findings.extend(_check_visible_output_health(entry))
        findings.extend(_check_possible_output_truncation(entry))
        if entry.visibility in {"public", "private"}:
            match = self._identity_pattern.search(entry.content)
            if match:
                findings.append(
                    MonitorFinding(
                        code="possible_identity_leak",
                        severity="error" if self.strict else "warning",
                        message="Participant may have referenced model identity or training details.",
                        speaker=entry.speaker,
                        turn_id=entry.turn_id,
                        evidence=match.group(0),
                    )
                )
        return findings

    def check_required_keys(
        self,
        entry: TranscriptEntry,
        parsed: dict[str, Any] | None,
        required_keys: list[str],
        require_json: bool = False,
    ) -> list[MonitorFinding]:
        if not required_keys and not require_json:
            return []
        if parsed is None:
            return [
                MonitorFinding(
                    code="missing_structured_json",
                    severity="error",
                    message="Required structured JSON was not parsed.",
                    speaker=entry.speaker,
                    turn_id=entry.turn_id,
                )
            ]
        missing = [key for key in required_keys if key not in parsed]
        if not missing:
            return []
        return [
            MonitorFinding(
                code="missing_required_keys",
                severity="error",
                message=f"Structured output is missing keys: {', '.join(missing)}",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
            )
        ]

    def check_structured_values(
        self,
        entry: TranscriptEntry,
        parsed: dict[str, Any] | None,
        participant_ids: list[str] | None = None,
    ) -> list[MonitorFinding]:
        if parsed is None:
            return []

        findings: list[MonitorFinding] = []
        expected_participants = set(participant_ids or [])
        participant_id = parsed.get("participant_id")
        if "participant_id" in parsed and participant_id != entry.speaker:
            findings.append(
                MonitorFinding(
                    code="participant_id_mismatch",
                    severity="error",
                    message="Structured participant_id must match the transcript speaker.",
                    speaker=entry.speaker,
                    turn_id=entry.turn_id,
                    evidence=str(participant_id),
                )
            )

        phase = parsed.get("phase")
        if "phase" in parsed and phase != entry.phase:
            findings.append(
                MonitorFinding(
                    code="phase_mismatch",
                    severity="error",
                    message="Structured phase must match the transcript phase.",
                    speaker=entry.speaker,
                    turn_id=entry.turn_id,
                    evidence=str(phase),
                )
            )

        round_index = parsed.get("round_index")
        if "round_index" in parsed and round_index != entry.round_index:
            findings.append(
                MonitorFinding(
                    code="round_index_mismatch",
                    severity="error",
                    message="Structured round_index must match the transcript round.",
                    speaker=entry.speaker,
                    turn_id=entry.turn_id,
                    evidence=str(round_index),
                )
            )

        findings.extend(_check_metadata_identity_fields(entry, parsed))
        findings.extend(_check_ranking_field(entry, parsed, "ranking", expected_participants))
        findings.extend(_check_ranking_field(entry, parsed, "current_ranking", expected_participants))
        findings.extend(_check_list_of_strings_fields(entry, parsed))
        findings.extend(_check_judge_score_fields(entry, parsed, expected_participants))
        findings.extend(_check_follow_up_fields(entry, parsed, expected_participants))
        findings.extend(_check_comparative_judgment_fields(entry, parsed, expected_participants))

        confidence = parsed.get("confidence")
        if "confidence" in parsed and (
            isinstance(confidence, bool)
            or not isinstance(confidence, Real)
            or not 0 <= float(confidence) <= 1
        ):
            findings.append(
                MonitorFinding(
                    code="invalid_confidence",
                    severity="error",
                    message="Structured confidence must be a number from 0 to 1.",
                    speaker=entry.speaker,
                    turn_id=entry.turn_id,
                    evidence=str(confidence),
                )
            )
        return findings


def _check_metadata_identity_fields(
    entry: TranscriptEntry,
    parsed: dict[str, Any],
) -> list[MonitorFinding]:
    checks = {
        "judge_id": "judge",
        "candidate_id": "candidate",
        "interviewer_id": "interviewer",
        "respondent_id": "respondent",
        "target_participant_id": "respondent",
        "probe_id": "probe_id",
    }
    findings: list[MonitorFinding] = []
    for parsed_key, metadata_key in checks.items():
        if parsed_key not in parsed or metadata_key not in entry.metadata:
            continue
        parsed_value = parsed.get(parsed_key)
        metadata_value = entry.metadata.get(metadata_key)
        if parsed_value != metadata_value:
            findings.append(
                MonitorFinding(
                    code=f"{parsed_key}_mismatch",
                    severity="error",
                    message=f"Structured {parsed_key} must match transcript metadata {metadata_key}.",
                    speaker=entry.speaker,
                    turn_id=entry.turn_id,
                    evidence=str(parsed_value),
                )
            )
    return findings


def _check_judge_score_fields(
    entry: TranscriptEntry,
    parsed: dict[str, Any],
    expected_participants: set[str],
) -> list[MonitorFinding]:
    findings: list[MonitorFinding] = []
    ability_score = parsed.get("ability_score")
    if "ability_score" in parsed and not _is_score(ability_score):
        findings.append(
            MonitorFinding(
                code="invalid_ability_score",
                severity="error",
                message="Structured ability_score must be a number from 0 to 100.",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
                evidence=str(ability_score),
            )
        )

    if "scores" not in parsed:
        return findings
    scores = parsed.get("scores")
    if not isinstance(scores, dict):
        findings.append(
            MonitorFinding(
                code="invalid_scores_shape",
                severity="error",
                message="Structured scores must map candidate IDs to numbers from 0 to 100.",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
                evidence=type(scores).__name__,
            )
        )
        return findings
    score_ids = {str(candidate_id) for candidate_id in scores}
    invalid_values = [str(candidate_id) for candidate_id, score in scores.items() if not _is_score(score)]
    if expected_participants and score_ids != expected_participants:
        findings.append(
            MonitorFinding(
                code="invalid_score_ids",
                severity="error",
                message="Structured scores must contain every candidate ID exactly once.",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
                evidence=", ".join(sorted(score_ids)),
            )
        )
    if invalid_values:
        findings.append(
            MonitorFinding(
                code="invalid_score_values",
                severity="error",
                message="Every structured score must be a number from 0 to 100.",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
                evidence=", ".join(invalid_values),
            )
        )
    return findings


def _check_follow_up_fields(
    entry: TranscriptEntry,
    parsed: dict[str, Any],
    expected_participants: set[str],
) -> list[MonitorFinding]:
    findings: list[MonitorFinding] = []
    follow_up = parsed.get("follow_up_candidates")
    if "follow_up_candidates" in parsed:
        if not isinstance(follow_up, list) or not all(isinstance(item, str) for item in follow_up):
            findings.append(
                MonitorFinding(
                    code="invalid_follow_up_candidates_shape",
                    severity="error",
                    message="Structured follow_up_candidates must be an array of candidate IDs.",
                    speaker=entry.speaker,
                    turn_id=entry.turn_id,
                )
            )
        else:
            invalid_ids = sorted(set(follow_up) - expected_participants)
            if invalid_ids or len(follow_up) != len(set(follow_up)):
                findings.append(
                    MonitorFinding(
                        code="invalid_follow_up_candidates",
                        severity="error",
                        message="Structured follow_up_candidates must contain unique valid candidate IDs.",
                        speaker=entry.speaker,
                        turn_id=entry.turn_id,
                        evidence=", ".join(invalid_ids),
                    )
                )

    uncertain_pairs = parsed.get("uncertain_pairs")
    if "uncertain_pairs" in parsed and (
        not isinstance(uncertain_pairs, list)
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or pair[0] == pair[1]
            or any(not isinstance(item, str) or item not in expected_participants for item in pair)
            for pair in uncertain_pairs
        )
    ):
        findings.append(
            MonitorFinding(
                code="invalid_uncertain_pairs",
                severity="error",
                message="Structured uncertain_pairs must contain valid two-candidate ID arrays.",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
            )
        )
    return findings


def _check_comparative_judgment_fields(
    entry: TranscriptEntry,
    parsed: dict[str, Any],
    expected_participants: set[str],
) -> list[MonitorFinding]:
    findings: list[MonitorFinding] = []
    respondents = entry.metadata.get("respondents")
    expected_respondents = (
        {str(value) for value in respondents}
        if isinstance(respondents, list)
        else set()
    )

    summaries = parsed.get("candidate_summaries")
    if "candidate_summaries" in parsed and (
        not isinstance(summaries, dict)
        or {str(value) for value in summaries} != expected_respondents
        or not all(isinstance(value, str) and value.strip() for value in summaries.values())
    ):
        findings.append(
            MonitorFinding(
                code="invalid_candidate_summaries",
                severity="error",
                message="candidate_summaries must map every routed candidate to non-empty text.",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
            )
        )

    ordering = parsed.get("ordering")
    if "ordering" in parsed and (
        not isinstance(ordering, list)
        or not all(isinstance(value, str) for value in ordering)
        or len(ordering) != len(set(ordering))
        or set(ordering) != expected_respondents
    ):
        findings.append(
            MonitorFinding(
                code="invalid_probe_ordering",
                severity="error",
                message="ordering must contain every routed candidate exactly once.",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
            )
        )

    ties = parsed.get("ties")
    if "ties" in parsed and (
        not isinstance(ties, list)
        or any(
            not isinstance(group, list)
            or len(group) < 2
            or len(group) != len(set(group))
            or any(not isinstance(value, str) or value not in expected_respondents for value in group)
            for group in ties
        )
    ):
        findings.append(
            MonitorFinding(
                code="invalid_probe_ties",
                severity="error",
                message="ties must contain groups of two or more routed candidate IDs.",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
            )
        )

    dossiers = parsed.get("candidate_dossiers")
    if "candidate_dossiers" in parsed and (
        not isinstance(dossiers, dict)
        or {str(value) for value in dossiers} != expected_participants
        or not all(isinstance(value, str) and value.strip() for value in dossiers.values())
    ):
        findings.append(
            MonitorFinding(
                code="invalid_candidate_dossiers",
                severity="error",
                message="candidate_dossiers must map every candidate to non-empty cumulative evidence.",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
            )
        )

    validity = parsed.get("probe_validity")
    if "probe_validity" in parsed and validity not in {"informative", "limited", "invalid"}:
        findings.append(
            MonitorFinding(
                code="invalid_probe_validity",
                severity="error",
                message="probe_validity must be informative, limited, or invalid.",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
                evidence=str(validity),
            )
        )
    return findings


def _is_score(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and 0 <= float(value) <= 100


def _check_visible_output_health(entry: TranscriptEntry) -> list[MonitorFinding]:
    usage = entry.metadata.get("usage", {})
    health = inspect_output_health(entry.content, usage if isinstance(usage, dict) else {})
    findings: list[MonitorFinding] = []
    if not health.has_visible_text:
        code = "empty_visible_output_after_reasoning" if health.reasoning_tokens else "empty_visible_output"
        findings.append(
            MonitorFinding(
                code=code,
                severity="error",
                message="Model returned no visible response content.",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
                evidence=format_output_health_evidence(health),
            )
        )
    if not health.reasoning_tokens:
        return findings
    if entry.metadata.get("parse_error") and health.reasoning_dominated:
        findings.append(
            MonitorFinding(
                code="reasoning_dominated_structured_output_failure",
                severity="error",
                message=(
                    "Required structured output failed while reasoning tokens dominated "
                    "the completion budget."
                ),
                speaker=entry.speaker,
                turn_id=entry.turn_id,
                evidence=format_output_health_evidence(health),
            )
        )
    return findings


def _check_possible_output_truncation(entry: TranscriptEntry) -> list[MonitorFinding]:
    if not entry.metadata.get("parse_error"):
        return []
    usage = entry.metadata.get("usage", {})
    request_params = entry.metadata.get("request_params", {})
    if not isinstance(usage, dict) or not isinstance(request_params, dict):
        return []
    completion_tokens = _optional_int(usage.get("completion_tokens"))
    max_tokens = _optional_int(request_params.get("max_tokens"))
    if completion_tokens is None or max_tokens is None:
        return []
    if completion_tokens < max_tokens:
        return []
    return [
        MonitorFinding(
            code="structured_output_may_be_truncated",
            severity="warning",
            message="Structured output failed parsing after reaching the configured max_tokens cap.",
            speaker=entry.speaker,
            turn_id=entry.turn_id,
            evidence=f"completion_tokens={completion_tokens}; max_tokens={max_tokens}",
        )
    ]


def _check_list_of_strings_fields(
    entry: TranscriptEntry,
    parsed: dict[str, Any],
) -> list[MonitorFinding]:
    findings: list[MonitorFinding] = []
    for field_name in sorted(LIST_OF_STRINGS_FIELDS):
        if field_name not in parsed:
            continue
        value = parsed[field_name]
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            continue
        findings.append(
            MonitorFinding(
                code=f"invalid_{field_name}_shape",
                severity="error",
                message=f"Structured {field_name} must be a JSON array of strings.",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
                evidence=type(value).__name__,
            )
        )
    return findings


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _check_ranking_field(
    entry: TranscriptEntry,
    parsed: dict[str, Any],
    field_name: str,
    expected_participants: set[str],
) -> list[MonitorFinding]:
    if field_name not in parsed:
        return []

    ranking = parsed.get(field_name)
    code_suffix = "ranking" if field_name == "ranking" else "current_ranking"
    if not is_supported_ranking_shape(ranking):
        return [
            MonitorFinding(
                code=f"invalid_{code_suffix}_shape",
                severity="error",
                message=(
                    f"Structured {field_name} must be an ordered list of participant IDs "
                    "or a participant-to-rank map."
                ),
                speaker=entry.speaker,
                turn_id=entry.turn_id,
            )
        ]

    findings: list[MonitorFinding] = []
    ranking_id_list = ranking_ids(ranking)
    duplicate_ids = sorted({item for item in ranking_id_list if ranking_id_list.count(item) > 1})
    duplicate_positions = duplicate_rank_positions(ranking)
    unknown_ids = sorted(set(ranking_id_list) - expected_participants) if expected_participants else []
    missing_ids = sorted(expected_participants - set(ranking_id_list)) if expected_participants else []
    if duplicate_ids:
        findings.append(
            MonitorFinding(
                code=f"duplicate_{code_suffix}_ids",
                severity="error",
                message=f"Structured {field_name} must not repeat participant IDs.",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
                evidence=", ".join(duplicate_ids),
            )
        )
    if duplicate_positions:
        findings.append(
            MonitorFinding(
                code=f"duplicate_{code_suffix}_positions",
                severity="error",
                message=f"Structured {field_name} must not assign the same rank position more than once.",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
                evidence=", ".join(duplicate_positions),
            )
        )
    if unknown_ids:
        findings.append(
            MonitorFinding(
                code=f"unknown_{code_suffix}_ids",
                severity="error",
                message=f"Structured {field_name} contains IDs that are not experiment participants.",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
                evidence=", ".join(unknown_ids),
            )
        )
    if missing_ids:
        findings.append(
            MonitorFinding(
                code=f"missing_{code_suffix}_ids",
                severity="error",
                message=f"Structured {field_name} must include every experiment participant.",
                speaker=entry.speaker,
                turn_id=entry.turn_id,
                evidence=", ".join(missing_ids),
            )
        )
    return findings
