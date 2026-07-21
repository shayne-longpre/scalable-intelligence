from __future__ import annotations

from collections import Counter
from typing import Any

from ai_council.taxonomy import EVALUATION_STRATEGY, QUESTION_TYPE, load_taxonomy, taxonomy_hits_for_entry


SCHEMA_VERSION = "2026-07-20"


def extract_posthoc_interactions(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract traceable interaction objects from a transcript.

    This is deliberately deterministic. It does not infer hidden structure from
    free text beyond lightweight candidate-question detection; richer labels can
    be added later on top of these stable turn-linked objects.
    """

    taxonomy = load_taxonomy()
    by_turn_id = {
        entry.get("turn_id"): entry
        for entry in entries
        if isinstance(entry.get("turn_id"), int)
    }
    assessments_by_answer = _assessments_by_answer(entries)

    qa_pairs = []
    seen_pair_ids: set[str] = set()
    for answer in entries:
        metadata = answer.get("metadata") or {}
        if not _is_answer_entry(metadata):
            continue
        question_turn_id = metadata.get("question_turn_id") or metadata.get("source_turn_id")
        if not isinstance(question_turn_id, int):
            continue
        question = by_turn_id.get(question_turn_id)
        if question is None:
            continue
        pair = _build_qa_pair(
            question=question,
            answer=answer,
            assessment=assessments_by_answer.get(answer.get("turn_id")),
            taxonomy=taxonomy,
        )
        if pair["pair_id"] in seen_pair_ids:
            continue
        seen_pair_ids.add(pair["pair_id"])
        qa_pairs.append(pair)

    discussion_turns = [
        _turn_record(entry, taxonomy)
        for entry in entries
        if _is_discussion_turn(entry)
    ]
    candidate_question_turns = [
        record
        for record in discussion_turns
        if _is_freeform_candidate_probe(record)
    ]
    probe_events = _probe_events(discussion_turns, candidate_question_turns)
    probe_comparisons = _structured_records(entries, "probe_comparison")
    wave_judgments = _structured_records(entries, "wave_judgment")
    adaptive_decisions = _adaptive_decisions(
        probe_events,
        probe_comparisons,
        wave_judgments,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "qa_pairs": qa_pairs,
        "probe_events": probe_events,
        "discussion_turns": discussion_turns,
        "candidate_question_turns": candidate_question_turns,
        "probe_comparisons": probe_comparisons,
        "wave_judgments": wave_judgments,
        "adaptive_decisions": adaptive_decisions,
        "summary": {
            "qa_pair_count": len(qa_pairs),
            "probe_event_count": len(probe_events),
            "discussion_turn_count": len(discussion_turns),
            "candidate_question_turn_count": len(candidate_question_turns),
            "probe_comparison_count": len(probe_comparisons),
            "wave_judgment_count": len(wave_judgments),
            "adaptive_decision_count": len(adaptive_decisions),
            "probe_validity_frequency": dict(
                Counter(
                    record.get("parsed", {}).get("probe_validity")
                    for record in probe_comparisons
                    if isinstance(record.get("parsed"), dict)
                )
            ),
            "qa_pairs_by_mode": dict(Counter(pair["interaction_mode"] for pair in qa_pairs)),
            "question_type_frequency": _tag_frequency(probe_events, "question_type_tags"),
            "strategy_frequency": _tag_frequency(probe_events, "strategy_tags"),
        },
    }


def build_probe_answer_archive(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve exact independent-judge probes and answers for later reanalysis."""
    questions = {
        entry.get("turn_id"): entry
        for entry in entries
        if (entry.get("metadata") or {}).get("interaction_role") == "question"
        and (entry.get("metadata") or {}).get("interaction_mode")
        == "independent_judge_ranking"
    }
    comparisons_by_probe = {
        (entry.get("metadata") or {}).get("probe_id"): entry.get("metadata") or {}
        for entry in entries
        if (entry.get("metadata") or {}).get("interaction_role") == "probe_comparison"
    }
    grouped: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        metadata = entry.get("metadata") or {}
        question_turn_id = metadata.get("question_turn_id")
        if (
            metadata.get("interaction_role") != "answer"
            or metadata.get("interaction_mode") != "independent_judge_ranking"
            or question_turn_id not in questions
        ):
            continue
        grouped.setdefault(question_turn_id, []).append(entry)

    probes = []
    for question_turn_id, question in sorted(questions.items()):
        question_metadata = question.get("metadata") or {}
        comparison_metadata = comparisons_by_probe.get(question_metadata.get("probe_id"), {})
        answers = []
        for answer in sorted(
            grouped.get(question_turn_id, []),
            key=lambda item: str((item.get("metadata") or {}).get("respondent", "")),
        ):
            metadata = answer.get("metadata") or {}
            answers.append(
                {
                    "candidate_id": metadata.get("respondent") or answer.get("speaker"),
                    "answer_turn_id": answer.get("turn_id"),
                    "answer_stream_id": metadata.get("stream_id"),
                    "model_ref": metadata.get("model_ref"),
                    "content": answer.get("content", ""),
                    "finish_reason": metadata.get("finish_reason"),
                    "answer_unavailable": bool(metadata.get("answer_unavailable")),
                    "usage": metadata.get("usage", {}),
                }
            )
        probes.append(
            {
                "judge_id": question_metadata.get("interviewer") or question.get("speaker"),
                "probe_id": question_metadata.get("probe_id"),
                "round_index": question.get("round_index"),
                "probe_sequence_number": question_metadata.get("probe_sequence_number"),
                "question_turn_id": question_turn_id,
                "question_stream_id": question_metadata.get("stream_id"),
                "question_text": question.get("content", ""),
                "answer_presentation_order": comparison_metadata.get(
                    "answer_presentation_order",
                    [],
                ),
                "comparison_order": comparison_metadata.get("comparison_order"),
                "comparison_seed": comparison_metadata.get("comparison_seed"),
                "answers": answers,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_count": len(probes),
        "answer_count": sum(len(probe["answers"]) for probe in probes),
        "probes": probes,
    }


def _structured_records(
    entries: list[dict[str, Any]],
    interaction_role: str,
) -> list[dict[str, Any]]:
    records = []
    for entry in entries:
        metadata = entry.get("metadata") or {}
        if metadata.get("interaction_role") != interaction_role:
            continue
        records.append(
            {
                "turn_id": entry.get("turn_id"),
                "phase": entry.get("phase"),
                "round_index": entry.get("round_index"),
                "speaker": entry.get("speaker"),
                "stream_id": metadata.get("stream_id"),
                "probe_id": metadata.get("probe_id"),
                "probe_sequence_number": metadata.get("probe_sequence_number"),
                "question_turn_id": metadata.get("question_turn_id"),
                "answer_turn_ids": metadata.get("answer_turn_ids", []),
                "probe_comparison_turn_ids": metadata.get(
                    "probe_comparison_turn_ids",
                    [],
                ),
                "prior_judgment_turn_id": metadata.get("prior_judgment_turn_id"),
                "judgment_probe_count": metadata.get("judgment_probe_count"),
                "respondents": metadata.get("respondents", []),
                "participants": metadata.get("participants", []),
                "parsed": entry.get("parsed"),
            }
        )
    return records


def _build_qa_pair(
    *,
    question: dict[str, Any],
    answer: dict[str, Any],
    assessment: dict[str, Any] | None,
    taxonomy: dict[str, Any],
) -> dict[str, Any]:
    question_metadata = question.get("metadata") or {}
    answer_metadata = answer.get("metadata") or {}
    assessment_metadata = assessment.get("metadata") if assessment else {}
    assessment_metadata = assessment_metadata or {}
    question_turn_id = question.get("turn_id")
    answer_turn_id = answer.get("turn_id")
    assessment_turn_id = assessment.get("turn_id") if assessment else None
    interaction_mode = (
        answer_metadata.get("interaction_mode")
        or question_metadata.get("interaction_mode")
        or _legacy_interaction_mode(answer_metadata)
    )
    respondent_id = answer_metadata.get("respondent") or answer.get("speaker")
    interviewer_id = (
        answer_metadata.get("interviewer")
        or question_metadata.get("interviewer")
        or answer_metadata.get("test_originator")
        or question.get("speaker")
    )
    question_hits = taxonomy_hits_for_entry(question, taxonomy)
    assessment_hits = taxonomy_hits_for_entry(assessment, taxonomy) if assessment else []
    question_strategies = _hits_by_dimension(question_hits, EVALUATION_STRATEGY)
    return {
        "pair_id": f"qa:{question_turn_id}:{answer_turn_id}",
        "interaction_mode": interaction_mode or "unknown",
        "stream_id": answer_metadata.get("stream_id") or question_metadata.get("stream_id"),
        "probe_id": answer_metadata.get("probe_id") or question_metadata.get("probe_id"),
        "round_index": answer.get("round_index") or question.get("round_index"),
        "interviewer_id": interviewer_id,
        "respondent_id": respondent_id,
        "question_turn_id": question_turn_id,
        "answer_turn_id": answer_turn_id,
        "assessment_turn_id": assessment_turn_id,
        "question_phase": question.get("phase"),
        "answer_phase": answer.get("phase"),
        "assessment_phase": assessment.get("phase") if assessment else None,
        "question_text": question.get("content", ""),
        "answer_text": answer.get("content", ""),
        "assessment_text": assessment.get("content", "") if assessment else "",
        "assessment_parsed": assessment.get("parsed") if assessment else None,
        "question_type_tags": _hits_by_dimension(question_hits, QUESTION_TYPE),
        "question_strategy_tags": question_strategies,
        "strategy_tags": question_strategies,
        "assessment_strategy_tags": _hits_by_dimension(
            assessment_hits,
            EVALUATION_STRATEGY,
        ),
        "metadata": {
            "question": _selected_metadata(question_metadata),
            "answer": _selected_metadata(answer_metadata),
            "assessment": _selected_metadata(assessment_metadata),
        },
    }


def _assessments_by_answer(entries: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    assessments: dict[int, dict[str, Any]] = {}
    for entry in entries:
        metadata = entry.get("metadata") or {}
        answer_turn_id = metadata.get("answer_turn_id")
        if isinstance(answer_turn_id, int):
            assessments.setdefault(answer_turn_id, entry)
            continue
        answer_turn_ids = metadata.get("answer_turn_ids")
        if isinstance(answer_turn_ids, list):
            for turn_id in answer_turn_ids:
                if isinstance(turn_id, int):
                    assessments.setdefault(turn_id, entry)
    return assessments


def _is_answer_entry(metadata: dict[str, Any]) -> bool:
    role = metadata.get("interaction_role")
    if role is not None:
        return role == "answer"
    return "source_turn_id" in metadata and "test_originator" in metadata and "respondent" in metadata


def _turn_record(entry: dict[str, Any], taxonomy: dict[str, Any]) -> dict[str, Any]:
    hits = taxonomy_hits_for_entry(entry, taxonomy)
    metadata = entry.get("metadata") or {}
    role = metadata.get("interaction_role")
    is_probe_text = role == "question" or (
        role == "discussion" and _looks_like_question(str(entry.get("content", "")))
    )
    is_evaluator_text = role != "answer"
    strategy_tags = _hits_by_dimension(hits, EVALUATION_STRATEGY) if is_evaluator_text else []
    if metadata.get("generation_stage") in {"baseline_battery", "baseline_probe"}:
        strategy_tags = [tag for tag in strategy_tags if tag["tag"] != "adaptive_followup"]
    return {
        "turn_id": entry.get("turn_id"),
        "phase": entry.get("phase"),
        "round_index": entry.get("round_index"),
        "speaker": entry.get("speaker"),
        "visibility": entry.get("visibility"),
        "interaction_mode": metadata.get("interaction_mode"),
        "interaction_role": role,
        "stream_id": metadata.get("stream_id"),
        "probe_id": metadata.get("probe_id"),
        "probe_number": metadata.get("probe_number"),
        "probe_count": metadata.get("probe_count"),
        "probe_sequence_number": metadata.get("probe_sequence_number"),
        "generation_stage": metadata.get("generation_stage"),
        "evidence_turn_ids_available": metadata.get("evidence_turn_ids_available", []),
        "prior_ranking_turn_id": metadata.get("prior_ranking_turn_id"),
        "respondents": metadata.get("respondents", []),
        "content": entry.get("content", ""),
        "question_type_tags": (
            _hits_by_dimension(hits, QUESTION_TYPE) if is_probe_text else []
        ),
        "strategy_tags": strategy_tags,
    }


def _is_discussion_turn(entry: dict[str, Any]) -> bool:
    metadata = entry.get("metadata") or {}
    if entry.get("visibility") == "public":
        return True
    return metadata.get("interaction_role") in {"question", "answer", "discussion"}


def _looks_like_question(value: str) -> bool:
    lowered = value.lower()
    return "?" in value or "probe" in lowered or "question" in lowered or "challenge" in lowered


def _is_freeform_candidate_probe(record: dict[str, Any]) -> bool:
    if not _looks_like_question(record["content"]):
        return False
    return record.get("visibility") == "public" or record.get("interaction_role") == "discussion"


def _probe_events(
    discussion_turns: list[dict[str, Any]],
    candidate_question_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates_by_turn = {turn.get("turn_id"): turn for turn in candidate_question_turns}
    events = []
    seen_turn_ids = set()
    for turn in discussion_turns:
        role = turn.get("interaction_role")
        is_routed_question = role == "question"
        is_candidate = turn.get("turn_id") in candidates_by_turn
        is_discussion_move = role == "discussion" and turn.get("visibility") == "public"
        if not is_routed_question and not is_candidate and not is_discussion_move:
            continue
        turn_id = turn.get("turn_id")
        if turn_id in seen_turn_ids:
            continue
        seen_turn_ids.add(turn_id)
        events.append(
            {
                "event_id": f"probe:{turn_id}",
                "turn_id": turn_id,
                "phase": turn.get("phase"),
                "round_index": turn.get("round_index"),
                "speaker": turn.get("speaker"),
                "visibility": turn.get("visibility"),
                "interaction_mode": turn.get("interaction_mode"),
                "interaction_role": role,
                "stream_id": turn.get("stream_id"),
                "probe_id": turn.get("probe_id"),
                "probe_number": turn.get("probe_number"),
                "probe_count": turn.get("probe_count"),
                "probe_sequence_number": turn.get("probe_sequence_number"),
                "generation_stage": turn.get("generation_stage"),
                "evidence_turn_ids_available": turn.get(
                    "evidence_turn_ids_available",
                    [],
                ),
                "prior_ranking_turn_id": turn.get("prior_ranking_turn_id"),
                "respondents": turn.get("respondents", []),
                "event_type": _probe_event_type(is_routed_question, is_candidate),
                "content": turn.get("content", ""),
                "question_type_tags": turn.get("question_type_tags", []),
                "strategy_tags": turn.get("strategy_tags", []),
            }
        )
    return events


def _adaptive_decisions(
    probe_events: list[dict[str, Any]],
    probe_comparisons: list[dict[str, Any]],
    wave_judgments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    comparisons_by_probe = {
        record.get("probe_id"): record
        for record in probe_comparisons
        if record.get("probe_id") is not None
    }
    judgments_by_turn = {
        record.get("turn_id"): record
        for record in wave_judgments
        if isinstance(record.get("turn_id"), int)
    }
    judgments_by_round = {
        (record.get("speaker"), record.get("round_index")): record
        for record in wave_judgments
    }
    comparisons_by_turn = {
        record.get("turn_id"): record
        for record in probe_comparisons
        if isinstance(record.get("turn_id"), int)
    }
    previous_targets_by_speaker: dict[str, list[str]] = {}
    decisions = []

    for event in sorted(probe_events, key=lambda item: item.get("turn_id") or 0):
        speaker = str(event.get("speaker"))
        actual = _string_list(event.get("respondents"))
        previous_targets = previous_targets_by_speaker.get(speaker, [])
        previous_targets_by_speaker[speaker] = actual
        if event.get("generation_stage") != "adaptive_followup":
            continue

        prior = judgments_by_turn.get(event.get("prior_ranking_turn_id"))
        prior_parsed = prior.get("parsed") if isinstance(prior, dict) else {}
        prior_parsed = prior_parsed if isinstance(prior_parsed, dict) else {}
        requested = _string_list(prior_parsed.get("follow_up_candidates"))
        uncertain_pairs = _id_pairs(prior_parsed.get("uncertain_pairs"))
        comparison = comparisons_by_probe.get(event.get("probe_id"))
        comparison_parsed = comparison.get("parsed") if isinstance(comparison, dict) else {}
        comparison_parsed = comparison_parsed if isinstance(comparison_parsed, dict) else {}
        result = judgments_by_round.get((event.get("speaker"), event.get("round_index")))
        result_parsed = result.get("parsed") if isinstance(result, dict) else {}
        result_parsed = result_parsed if isinstance(result_parsed, dict) else {}
        prior_ranking = _string_list(prior_parsed.get("ranking"))
        resulting_ranking = _string_list(result_parsed.get("ranking"))
        prior_confidence = _number(prior_parsed.get("confidence"))
        resulting_confidence = _number(result_parsed.get("confidence"))
        prior_comparison_ids = (
            prior.get("probe_comparison_turn_ids", []) if isinstance(prior, dict) else []
        )
        prior_validities = []
        for turn_id in prior_comparison_ids:
            record = comparisons_by_turn.get(turn_id)
            parsed = record.get("parsed") if isinstance(record, dict) else {}
            validity = parsed.get("probe_validity") if isinstance(parsed, dict) else None
            if isinstance(validity, str):
                prior_validities.append(validity)

        decisions.append(
            {
                "question_turn_id": event.get("turn_id"),
                "probe_id": event.get("probe_id"),
                "probe_sequence_number": event.get("probe_sequence_number"),
                "round_index": event.get("round_index"),
                "judge_id": event.get("speaker"),
                "prior_judgment_turn_id": event.get("prior_ranking_turn_id"),
                "actual_candidates": actual,
                "requested_candidates": requested,
                "selection_matches_request": (
                    set(actual) == set(requested) if requested else None
                ),
                "retained_candidates": [value for value in actual if value in previous_targets],
                "added_candidates": [value for value in actual if value not in previous_targets],
                "dropped_candidates": [value for value in previous_targets if value not in actual],
                "prior_uncertain_pairs": uncertain_pairs,
                "covered_uncertain_pairs": [
                    pair for pair in uncertain_pairs if set(pair).issubset(actual)
                ],
                "follow_up_rationale": _string_list(
                    prior_parsed.get("follow_up_rationale")
                ),
                "planned_strategy": _string_list(prior_parsed.get("next_probe_strategy")),
                "prior_round_probe_validities": prior_validities,
                "probe_validity": comparison_parsed.get("probe_validity"),
                "comparison_confidence": _number(comparison_parsed.get("confidence")),
                "prior_ranking": prior_ranking,
                "resulting_ranking": resulting_ranking,
                "ranking_changed": (
                    prior_ranking != resulting_ranking
                    if prior_ranking and resulting_ranking
                    else None
                ),
                "prior_confidence": prior_confidence,
                "resulting_confidence": resulting_confidence,
                "confidence_delta": (
                    resulting_confidence - prior_confidence
                    if prior_confidence is not None and resulting_confidence is not None
                    else None
                ),
            }
        )
    return decisions


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int))]


def _id_pairs(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    return [
        [str(pair[0]), str(pair[1])]
        for pair in value
        if isinstance(pair, list) and len(pair) == 2
    ]


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _probe_event_type(is_routed_question: bool, is_candidate: bool) -> str:
    if is_routed_question:
        return "routed_question"
    if is_candidate:
        return "candidate_question"
    return "discussion_move"


def _hits_by_dimension(hits: list[dict[str, Any]], dimension: str) -> list[dict[str, Any]]:
    return [
        {
            "tag": hit["tag"],
            "label": hit.get("label", hit["tag"]),
            "matched_indicators": hit.get("matched_indicators", []),
        }
        for hit in hits
        if hit.get("dimension", EVALUATION_STRATEGY) == dimension
    ]


def _tag_frequency(items: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter()
    for item in items:
        for tag in item.get(field, []):
            counts[tag["tag"]] += 1
    return dict(counts)


def _legacy_interaction_mode(metadata: dict[str, Any]) -> str | None:
    if "test_originator" in metadata:
        return "public_test_matrix"
    return None


def _selected_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "interaction_mode",
        "interaction_role",
        "stream_id",
        "probe_id",
        "interviewer",
        "respondent",
        "respondents",
        "test_originator",
        "source_phase",
        "source_turn_id",
        "question_turn_id",
        "answer_turn_id",
        "answer_turn_ids",
        "probe_number",
        "probe_count",
        "probe_sequence_number",
        "probe_schedule",
        "generation_stage",
        "evidence_turn_ids_available",
        "prior_ranking_turn_id",
        "prior_judgment_turn_id",
        "probe_comparison_turn_ids",
        "judgment_probe_count",
        "judgment_probe_total",
        "is_primary_judgment",
        "parse_error",
        "finish_reason",
        "model_ref",
        "model",
    ]
    return {key: metadata[key] for key in keys if key in metadata}
