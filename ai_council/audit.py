from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any


SCHEMA_VERSION = "2026-07-20.3"

STOPWORDS = {
    "about",
    "after",
    "again",
    "answer",
    "because",
    "before",
    "being",
    "between",
    "could",
    "from",
    "have",
    "into",
    "more",
    "must",
    "only",
    "other",
    "participant",
    "participants",
    "probe",
    "question",
    "respondent",
    "should",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "this",
    "turn",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
}


def audit_experiment_behavior(
    entries: list[dict[str, Any]],
    extraction: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return deterministic warnings about likely protocol confusion.

    The audit is intentionally conservative and trace-linked. It should help
    inspect runs; it should not be treated as a ground-truth behavioral label.
    """

    findings: list[dict[str, Any]] = []
    findings.extend(_structured_json_findings(entries))
    findings.extend(_qa_findings(extraction))
    findings.extend(_run_quality_findings(entries, extraction, config or {}))

    counts = Counter(finding["code"] for finding in findings)
    severities = Counter(finding["severity"] for finding in findings)
    return {
        "schema_version": SCHEMA_VERSION,
        "findings": findings,
        "summary": {
            "finding_count": len(findings),
            "codes": dict(counts),
            "severities": dict(severities),
        },
    }


def _structured_json_findings(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    for entry in entries:
        metadata = entry.get("metadata") or {}
        visible_retry = metadata.get("visible_text_retry")
        if isinstance(visible_retry, dict) and visible_retry.get("attempted"):
            recovered = bool(visible_retry.get("recovered"))
            reason = str(visible_retry.get("reason") or "empty")
            prefix = "truncated_output" if reason == "truncated" else "empty_visible_output"
            findings.append(
                _finding(
                    f"{prefix}_recovered" if recovered else f"{prefix}_unrecovered",
                    "info" if recovered else "error",
                    entry,
                    "A model call returned incomplete visible output and required a bounded retry.",
                    evidence=str(visible_retry.get("original_finish_reason") or ""),
                )
            )
        fallback = metadata.get("structured_json_fallback")
        if isinstance(fallback, dict) and fallback.get("applied"):
            findings.append(
                _finding(
                    "structured_json_fallback_applied",
                    "warning",
                    entry,
                    "Structured JSON used a deterministic fallback after model repair failed.",
                    evidence=str(fallback.get("failed_error") or ""),
                )
            )
        repair = metadata.get("structured_json_repair")
        if isinstance(repair, dict):
            findings.append(
                _finding(
                    "structured_json_repaired" if repair.get("repaired") else "structured_json_repair_failed",
                    "info" if repair.get("repaired") else "error",
                    entry,
                    "Private structured JSON required a same-model repair attempt.",
                    evidence=str(repair.get("original_structured_error") or repair.get("original_parse_error") or ""),
                )
            )
        elif metadata.get("parse_error"):
            findings.append(
                _finding(
                    "unrepaired_structured_json_parse_error",
                    "error",
                    entry,
                    "Structured JSON could not be parsed and was not repaired.",
                    evidence=str(metadata.get("parse_error")),
                )
            )
    return findings


def _run_quality_findings(
    entries: list[dict[str, Any]],
    extraction: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    findings.extend(_expected_qa_findings(entries, extraction))
    findings.extend(_memory_update_consistency_findings(entries))
    findings.extend(_final_ranking_findings(entries, config))
    findings.extend(_taxonomy_extraction_findings(extraction))
    findings.extend(_repair_rate_findings(entries))
    findings.extend(_completion_quality_findings(entries))
    return findings


def _expected_qa_findings(
    entries: list[dict[str, Any]],
    extraction: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = 0
    for entry in entries:
        metadata = entry.get("metadata") or {}
        if metadata.get("interaction_mode") != "round_robin_probes":
            continue
        if metadata.get("interaction_role") != "question":
            continue
        respondents = metadata.get("respondents")
        if isinstance(respondents, list):
            expected += len(respondents)
    if expected == 0:
        return []
    observed = len(extraction.get("qa_pairs", [])) if isinstance(extraction.get("qa_pairs"), list) else 0
    if observed == expected:
        return []
    return [
        _run_finding(
            "expected_observed_qa_mismatch",
            "error",
            "Structured run produced a different number of routed Q/A pairs than expected.",
            evidence=f"expected={expected}; observed={observed}",
        )
    ]


def _memory_update_consistency_findings(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected_by_probe: dict[str, set[str]] = {}
    for entry in entries:
        metadata = entry.get("metadata") or {}
        if metadata.get("interaction_mode") != "round_robin_probes":
            continue
        if metadata.get("interaction_role") != "question":
            continue
        probe_id = metadata.get("probe_id")
        respondents = metadata.get("respondents")
        if probe_id is None or not isinstance(respondents, list):
            continue
        expected_by_probe[str(probe_id)] = {str(respondent) for respondent in respondents}

    findings: list[dict[str, Any]] = []
    for entry in entries:
        metadata = entry.get("metadata") or {}
        if metadata.get("interaction_mode") != "round_robin_probes":
            continue
        if metadata.get("interaction_role") != "memory_update":
            continue
        expected = expected_by_probe.get(str(metadata.get("probe_id")))
        if not expected:
            continue
        parsed = entry.get("parsed")
        summaries = parsed.get("qa_assessment_summaries") if isinstance(parsed, dict) else None
        if not isinstance(summaries, list):
            continue
        observed = {
            str(summary.get("respondent_id"))
            for summary in summaries
            if isinstance(summary, dict) and summary.get("respondent_id") is not None
        }
        unexpected = sorted(observed - expected)
        missing = sorted(expected - observed)
        if unexpected:
            findings.append(
                _finding(
                    "memory_update_unexpected_respondent",
                    "warning",
                    entry,
                    "Memory update summarized a respondent who was not routed this probe.",
                    evidence=f"expected={sorted(expected)}; observed={sorted(observed)}; unexpected={unexpected}",
                )
            )
        if missing:
            findings.append(
                _finding(
                    "memory_update_missing_respondent",
                    "warning",
                    entry,
                    "Memory update omitted a respondent who was routed this probe.",
                    evidence=f"expected={sorted(expected)}; observed={sorted(observed)}; missing={missing}",
                )
            )
    return findings


def _final_ranking_findings(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    participant_ids = [
        str(participant.get("id"))
        for participant in config.get("participants", [])
        if isinstance(participant, dict) and participant.get("id") is not None
    ]
    if not participant_ids:
        return []
    phases = config.get("protocol", {}).get("phases", [])
    independent_judges = any(
        isinstance(phase, dict) and phase.get("kind") == "independent_judge_ranking"
        for phase in phases if isinstance(phases, list)
    )
    if independent_judges:
        expected_speakers = [
            str(judge.get("id"))
            for judge in config.get("judges", [])
            if isinstance(judge, dict) and judge.get("id") is not None
        ]
        final_entries = [
            entry
            for entry in entries
            if (entry.get("metadata") or {}).get("interaction_role")
            in {"judge_ranking", "wave_judgment"}
        ]
    else:
        expected_speakers = participant_ids
        final_entries = [
            entry
            for entry in entries
            if entry.get("phase") == "final_judgment" and entry.get("visibility") == "private"
        ]
    final_by_speaker = {str(entry.get("speaker")): entry for entry in final_entries}
    findings: list[dict[str, Any]] = []
    for speaker_id in expected_speakers:
        entry = final_by_speaker.get(speaker_id)
        if entry is None:
            findings.append(
                _run_finding(
                    "missing_final_judgment",
                    "error",
                    "Expected evaluator is missing a final ranking.",
                    evidence=f"evaluator={speaker_id}",
                )
            )
            continue
        parsed = entry.get("parsed")
        ranking = parsed.get("ranking") if isinstance(parsed, dict) else None
        if not isinstance(ranking, list) or set(str(item) for item in ranking) != set(participant_ids):
            findings.append(
                _finding(
                    "invalid_final_ranking",
                    "error",
                    entry,
                    "Final judgment did not rank every participant exactly once.",
                    evidence=f"ranking={ranking!r}",
                )
            )
        evidence = None
        if isinstance(parsed, dict):
            evidence = parsed.get("evidence", parsed.get("comparative_evidence"))
        if not isinstance(evidence, list) or not any(str(item).strip() for item in evidence):
            findings.append(
                _finding(
                    "final_judgment_missing_evidence",
                    "warning",
                    entry,
                    "Final judgment does not include usable evidence strings.",
                )
            )
    return findings


def _taxonomy_extraction_findings(extraction: dict[str, Any]) -> list[dict[str, Any]]:
    summary = extraction.get("summary", {}) if isinstance(extraction, dict) else {}
    probe_event_count = int(summary.get("probe_event_count") or 0)
    if probe_event_count == 0:
        return [
            _run_finding(
                "no_probe_events_detected",
                "warning",
                "No probe or discussion events were extracted for evolution analysis.",
            )
        ]
    question_types = summary.get("question_type_frequency")
    if isinstance(question_types, dict) and question_types:
        return []
    return [
        _run_finding(
            "no_question_type_labels",
            "warning",
            "Probe events were found, but no question-type labels were extracted.",
        )
    ]


def _repair_rate_findings(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    structured_entries = [
        entry
        for entry in entries
        if isinstance(entry.get("metadata"), dict)
        and (
            entry["metadata"].get("parse_error")
            or entry["metadata"].get("structured_error")
            or entry["metadata"].get("structured_json_repair")
            or entry["metadata"].get("structured_json_fallback")
        )
    ]
    repaired = [
        entry
        for entry in entries
        if isinstance(entry.get("metadata"), dict)
        and isinstance(entry["metadata"].get("structured_json_repair"), dict)
    ]
    if len(repaired) < 3:
        return []
    structured_total = max(len(structured_entries), len(repaired))
    rate = len(repaired) / structured_total if structured_total else 0.0
    if rate < 0.2:
        return []
    return [
        _run_finding(
            "high_structured_json_repair_rate",
            "warning",
            "Many structured turns required JSON repair; downstream rankings may be less reliable.",
            evidence=f"repairs={len(repaired)}; structured_error_related_turns={structured_total}; rate={rate:.2f}",
        )
    ]


def _completion_quality_findings(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    for entry in entries:
        metadata = entry.get("metadata") or {}
        if metadata.get("finish_reason") != "length":
            continue
        findings.append(
            _finding(
                "completion_truncated",
                "warning",
                entry,
                "Model response hit the configured token limit and may be incomplete.",
                evidence=f"role={metadata.get('interaction_role')}; max_tokens={_max_tokens(metadata)}",
            )
        )
    return findings


def _qa_findings(extraction: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    qa_pairs = extraction.get("qa_pairs", [])
    if not isinstance(qa_pairs, list):
        return findings

    findings.extend(_thin_answer_findings(qa_pairs))
    findings.extend(_possible_wrong_question_findings(qa_pairs, _question_sources(extraction)))
    findings.extend(_repeated_probe_findings(qa_pairs))
    return findings


def _thin_answer_findings(qa_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    for pair in qa_pairs:
        answer_word_count = _word_count(pair.get("answer_text", ""))
        if answer_word_count >= 20:
            continue
        findings.append(
            _pair_finding(
                "thin_answer",
                "warning",
                pair,
                "Answer is unusually short for a diagnostic probe.",
                evidence=f"{answer_word_count} words",
            )
        )
    return findings


def _possible_wrong_question_findings(
    qa_pairs: list[dict[str, Any]],
    question_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    findings = []
    question_tokens_by_turn = {
        source.get("question_turn_id"): _tokens(source.get("question_text", ""))
        for source in question_sources
    }
    question_text_by_turn = {
        source.get("question_turn_id"): source.get("question_text", "")
        for source in question_sources
    }
    for pair in qa_pairs:
        answer_tokens = _tokens(pair.get("answer_text", ""))
        if len(answer_tokens) < 12:
            continue
        actual_turn_id = pair.get("question_turn_id")
        actual_score = _overlap_count(answer_tokens, question_tokens_by_turn.get(actual_turn_id, set()))
        best_other_turn_id = None
        best_other_score = 0
        for other in question_sources:
            other_turn_id = other.get("question_turn_id")
            if other_turn_id == actual_turn_id:
                continue
            if other.get("round_index") != pair.get("round_index"):
                continue
            if other.get("interviewer_id") != pair.get("interviewer_id"):
                continue
            score = _overlap_count(answer_tokens, question_tokens_by_turn.get(other_turn_id, set()))
            if score > best_other_score:
                best_other_score = score
                best_other_turn_id = other_turn_id
        if best_other_turn_id is None or best_other_score < max(5, actual_score + 3):
            continue
        findings.append(
            _pair_finding(
                "possible_wrong_question_answered",
                "warning",
                pair,
                "Answer has stronger lexical overlap with a different same-interviewer probe.",
                evidence=(
                    f"actual_overlap={actual_score}; best_other_turn={best_other_turn_id}; "
                    f"best_other_overlap={best_other_score}; other_probe="
                    f"{_excerpt(question_text_by_turn.get(best_other_turn_id, ''))}"
                ),
            )
        )
    return findings


def _question_sources(extraction: dict[str, Any]) -> list[dict[str, Any]]:
    sources = []
    seen = set()
    for pair in extraction.get("qa_pairs", []):
        if not isinstance(pair, dict):
            continue
        question_turn_id = pair.get("question_turn_id")
        if question_turn_id in seen:
            continue
        seen.add(question_turn_id)
        sources.append(
            {
                "question_turn_id": question_turn_id,
                "round_index": pair.get("round_index"),
                "interviewer_id": pair.get("interviewer_id"),
                "question_text": pair.get("question_text", ""),
            }
        )
    for turn in extraction.get("discussion_turns", []):
        if not isinstance(turn, dict):
            continue
        if turn.get("interaction_role") != "question":
            continue
        question_turn_id = turn.get("turn_id")
        if question_turn_id in seen:
            continue
        seen.add(question_turn_id)
        sources.append(
            {
                "question_turn_id": question_turn_id,
                "round_index": turn.get("round_index"),
                "interviewer_id": turn.get("speaker"),
                "question_text": turn.get("content", ""),
            }
        )
    return sources


def _repeated_probe_findings(qa_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    seen_by_interviewer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    questions_by_pair = _unique_question_pairs(qa_pairs)
    for pair in questions_by_pair:
        interviewer = str(pair.get("interviewer_id"))
        tokens = _tokens(pair.get("question_text", ""))
        if len(tokens) < 8:
            seen_by_interviewer[interviewer].append(pair)
            continue
        for previous in seen_by_interviewer[interviewer]:
            previous_tokens = _tokens(previous.get("question_text", ""))
            if _jaccard(tokens, previous_tokens) < 0.75:
                continue
            findings.append(
                _pair_finding(
                    "repeated_probe",
                    "info",
                    pair,
                    "Interviewer repeated a highly similar probe.",
                    evidence=f"previous_question_turn={previous.get('question_turn_id')}",
                )
            )
            break
        seen_by_interviewer[interviewer].append(pair)
    return findings


def _unique_question_pairs(qa_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for pair in sorted(
        qa_pairs,
        key=lambda item: (
            item.get("round_index") or 0,
            item.get("question_turn_id") or 0,
            item.get("answer_turn_id") or 0,
        ),
    ):
        question_turn_id = pair.get("question_turn_id")
        if question_turn_id in seen:
            continue
        seen.add(question_turn_id)
        unique.append(pair)
    return unique


def _tokens(value: Any) -> set[str]:
    text = str(value).lower()
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9_'-]{2,}", text)
        if len(token) >= 4 and token not in STOPWORDS
    }


def _word_count(value: Any) -> int:
    return len(re.findall(r"\b\w+(?:['-]\w+)*\b", str(value)))


def _overlap_count(left: set[str], right: set[str]) -> int:
    return len(left & right)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _finding(
    code: str,
    severity: str,
    entry: dict[str, Any],
    message: str,
    *,
    evidence: str = "",
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "turn_id": entry.get("turn_id"),
        "speaker": entry.get("speaker"),
        "phase": entry.get("phase"),
        "evidence": evidence,
    }


def _pair_finding(
    code: str,
    severity: str,
    pair: dict[str, Any],
    message: str,
    *,
    evidence: str = "",
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "pair_id": pair.get("pair_id"),
        "question_turn_id": pair.get("question_turn_id"),
        "answer_turn_id": pair.get("answer_turn_id"),
        "interviewer_id": pair.get("interviewer_id"),
        "respondent_id": pair.get("respondent_id"),
        "round_index": pair.get("round_index"),
        "evidence": evidence,
    }


def _run_finding(
    code: str,
    severity: str,
    message: str,
    *,
    evidence: str = "",
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "evidence": evidence,
    }


def _excerpt(value: Any, limit: int = 120) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _max_tokens(metadata: dict[str, Any]) -> Any:
    request_params = metadata.get("request_params")
    if not isinstance(request_params, dict):
        return None
    return request_params.get("max_tokens")
