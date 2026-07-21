from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ai_council.rankings import kendall_tau_between, ranking_ids


_NON_TOPICAL_QUESTION_TYPES = {
    "instruction_following_format_control",
    "processing_speed_efficiency",
}
_QUESTION_TYPE_FAMILIES = {
    "coding_algorithmic_reasoning": "coding",
    "software_engineering_agentic": "coding",
    "dimensional_unit_reasoning": "quantitative_science",
    "quantitative_math_reasoning": "quantitative_science",
    "metacognitive_calibration_probe": "metacognition",
    "recursive_self_bias_probe": "metacognition",
}


def compute_ranking_metrics(rankings_by_phase: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    snapshots = _ranking_snapshots(rankings_by_phase)
    final_snapshots = _final_snapshots(snapshots)
    trajectory_snapshots = [
        item for item in snapshots if item.get("is_primary_judgment") is not False
    ]
    return {
        "snapshot_count": len(snapshots),
        "final_rankings": final_snapshots,
        "final_agreement": _agreement_summary(final_snapshots),
        "phase_agreement": _phase_agreement(snapshots),
        "churn_by_speaker": _churn_by_speaker(trajectory_snapshots),
        "probe_budget_rankings": [
            item for item in snapshots if item.get("judgment_probe_count") is not None
        ],
    }


def compute_evolution_metrics(extraction: dict[str, Any]) -> dict[str, Any]:
    events = extraction.get("probe_events", [])
    if not isinstance(events, list):
        events = []
    annotated = _annotate_probe_events(events)
    transition_counts = Counter(event.get("transition_label") for event in annotated)
    transition_counts.pop(None, None)
    topical_counts = Counter(event.get("topical_transition") for event in annotated)
    topical_counts.pop(None, None)
    dependency_counts = Counter(event.get("dependency_semantics") for event in annotated)
    dependency_counts.pop(None, None)
    return {
        "probe_event_count": len(annotated),
        "transition_counts": dict(transition_counts),
        "topical_transition_counts": dict(topical_counts),
        "dependency_counts": dict(dependency_counts),
        "question_types_by_round": _tag_counts_by_round(annotated, "question_type_tags"),
        "strategies_by_round": _tag_counts_by_round(annotated, "strategy_tags"),
        "probe_events": annotated,
        "adaptive": _adaptive_metrics(
            extraction.get("adaptive_decisions", []),
            annotated,
        ),
    }


def _adaptive_metrics(
    decisions: Any,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(decisions, list):
        decisions = []
    event_by_turn = {
        event.get("turn_id"): event
        for event in events
        if isinstance(event.get("turn_id"), int)
    }
    validity_counts: Counter[str] = Counter()
    target_change_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    trace = []
    confidence_deltas = []

    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        validity = decision.get("probe_validity")
        if isinstance(validity, str):
            validity_counts[validity] += 1
        target_change = _target_change(decision)
        target_change_counts[target_change] += 1
        outcome = _evidence_outcome(validity, decision.get("ranking_changed"))
        outcome_counts[outcome] += 1
        delta = decision.get("confidence_delta")
        if isinstance(delta, (int, float)):
            confidence_deltas.append(float(delta))
        event = event_by_turn.get(decision.get("question_turn_id"), {})
        trace.append(
            {
                **decision,
                "target_change": target_change,
                "evidence_outcome": outcome,
                "transition_label": event.get("transition_label"),
                "topical_transition": event.get("topical_transition"),
                "primary_question_type": event.get("primary_question_type"),
                "question_types": [
                    tag.get("tag")
                    for tag in event.get("question_type_tags", [])
                    if isinstance(tag, dict)
                    and isinstance(tag.get("tag"), str)
                    and tag.get("tag") not in _NON_TOPICAL_QUESTION_TYPES
                ],
            }
        )

    requested = [
        decision
        for decision in decisions
        if isinstance(decision, dict)
        and decision.get("selection_matches_request") is not None
    ]
    uncertain = [
        decision
        for decision in decisions
        if isinstance(decision, dict) and decision.get("prior_uncertain_pairs")
    ]
    return {
        "decision_count": len(trace),
        "selection_match_count": sum(
            decision.get("selection_matches_request") is True for decision in requested
        ),
        "selection_check_count": len(requested),
        "uncertainty_coverage_count": sum(
            bool(decision.get("covered_uncertain_pairs")) for decision in uncertain
        ),
        "uncertainty_check_count": len(uncertain),
        "target_change_counts": dict(target_change_counts),
        "probe_validity_counts": dict(validity_counts),
        "evidence_outcome_counts": dict(outcome_counts),
        "ranking_change_count": sum(
            decision.get("ranking_changed") is True
            for decision in decisions
            if isinstance(decision, dict)
        ),
        "mean_confidence_delta": _mean(confidence_deltas),
        "decision_trace": trace,
    }


def _target_change(decision: dict[str, Any]) -> str:
    added = bool(decision.get("added_candidates"))
    dropped = bool(decision.get("dropped_candidates"))
    if added and dropped:
        return "changed"
    if added:
        return "broadened"
    if dropped:
        return "narrowed"
    return "stable"


def _evidence_outcome(validity: Any, ranking_changed: Any) -> str:
    if validity == "invalid":
        return "invalid"
    if validity == "limited":
        return "limited"
    if validity == "informative" and ranking_changed is True:
        return "rank_changing"
    if validity == "informative":
        return "corroborating"
    return "unclassified"


def _ranking_snapshots(rankings_by_phase: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    snapshots = []
    index = 0
    for phase, rankings in rankings_by_phase.items():
        if not isinstance(rankings, list):
            continue
        for item in rankings:
            ranking = ranking_ids(item.get("ranking"), accept_id_objects=True)
            if not ranking:
                continue
            index += 1
            snapshots.append(
                {
                    "index": index,
                    "phase": phase,
                    "speaker": item.get("speaker"),
                    "round_index": item.get("round_index"),
                    "ranking": ranking,
                    "scores": item.get("scores"),
                    "confidence": item.get("confidence"),
                    "ranking_field": item.get("ranking_field"),
                    "judgment_probe_count": item.get("judgment_probe_count"),
                    "judgment_probe_total": item.get("judgment_probe_total"),
                    "is_primary_judgment": item.get("is_primary_judgment"),
                }
            )
    return snapshots


def _final_snapshots(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final = [item for item in snapshots if item.get("phase") == "final_judgment"]
    if final:
        return final
    primary = [item for item in snapshots if item.get("is_primary_judgment") is True]
    if primary:
        snapshots = primary
    latest_by_speaker = {}
    for item in snapshots:
        latest_by_speaker[item.get("speaker")] = item
    return list(latest_by_speaker.values())


def _agreement_summary(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    top1_counts = Counter()
    exact_counts = Counter()
    for item in snapshots:
        ranking = item.get("ranking") or []
        if ranking:
            top1_counts[ranking[0]] += 1
            exact_counts[" > ".join(ranking)] += 1
    for i, left in enumerate(snapshots):
        for right in snapshots[i + 1 :]:
            if left.get("speaker") == right.get("speaker"):
                continue
            tau = kendall_tau_between(left.get("ranking", []), right.get("ranking", []))
            pairs.append(
                {
                    "left_speaker": left.get("speaker"),
                    "right_speaker": right.get("speaker"),
                    "kendall_tau": tau,
                    "same_top1": _top1(left) == _top1(right),
                    "exact_match": left.get("ranking") == right.get("ranking"),
                }
            )
    taus = [pair["kendall_tau"] for pair in pairs if isinstance(pair.get("kendall_tau"), (int, float))]
    return {
        "pair_count": len(pairs),
        "mean_pairwise_tau": _mean(taus),
        "same_top1_pairs": sum(1 for pair in pairs if pair.get("same_top1")),
        "exact_match_pairs": sum(1 for pair in pairs if pair.get("exact_match")),
        "top1_counts": dict(top1_counts),
        "exact_ranking_counts": dict(exact_counts),
        "pairs": pairs,
    }


def _phase_agreement(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in snapshots:
        key_parts = [str(item.get("phase"))]
        if item.get("round_index") is not None:
            key_parts.append(f"round_{item['round_index']}")
        if item.get("judgment_probe_count") is not None:
            key_parts.append(f"probes_{item['judgment_probe_count']}")
        by_phase[":".join(key_parts)].append(item)
    return {
        phase: _agreement_summary(items)
        for phase, items in by_phase.items()
        if len(items) >= 2
    }


def _churn_by_speaker(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in snapshots:
        by_speaker[str(item.get("speaker"))].append(item)
    churn = {}
    for speaker, items in by_speaker.items():
        transitions = []
        for previous, current in zip(items, items[1:]):
            tau = kendall_tau_between(previous.get("ranking", []), current.get("ranking", []))
            transitions.append(
                {
                    "from_phase": previous.get("phase"),
                    "to_phase": current.get("phase"),
                    "from_round": previous.get("round_index"),
                    "to_round": current.get("round_index"),
                    "kendall_tau": tau,
                    "top1_changed": _top1(previous) != _top1(current),
                    "previous": previous.get("ranking"),
                    "current": current.get("ranking"),
                }
            )
        churn[speaker] = {
            "snapshot_count": len(items),
            "transition_count": len(transitions),
            "mean_adjacent_tau": _mean(
                transition["kendall_tau"]
                for transition in transitions
                if isinstance(transition.get("kendall_tau"), (int, float))
            ),
            "top1_changes": sum(1 for transition in transitions if transition.get("top1_changed")),
            "transitions": transitions,
        }
    return churn


def _annotate_probe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous_by_speaker: dict[str, dict[str, Any]] = {}
    annotated = []
    for event in sorted(events, key=lambda item: item.get("turn_id") or 0):
        speaker = str(event.get("speaker"))
        previous = previous_by_speaker.get(speaker)
        topical_transition = _topical_transition(previous, event)
        dependency_semantics = _dependency_semantics(event)
        label = _transition_label(topical_transition, dependency_semantics)
        record = {
            **event,
            "transition_label": label,
            "topical_transition": topical_transition,
            "dependency_semantics": dependency_semantics,
            "primary_question_type": _primary_question_type(
                event.get("question_type_tags", [])
            ),
            "primary_strategy": _primary_tag(event.get("strategy_tags", [])),
        }
        annotated.append(record)
        previous_by_speaker[speaker] = event
    return annotated


def _topical_transition(previous: dict[str, Any] | None, current: dict[str, Any]) -> str:
    if previous is None:
        return "opening_probe"
    previous_question_types = _topic_signature(previous.get("question_type_tags", []))
    current_question_types = _topic_signature(current.get("question_type_tags", []))
    previous_strategies = _tag_set(previous.get("strategy_tags", []))
    current_strategies = _tag_set(current.get("strategy_tags", []))
    if not previous_question_types or not current_question_types:
        return "topic_unresolved"
    new_types = current_question_types - previous_question_types
    if current_question_types & previous_question_types:
        return "same_area_new_angle"
    if new_types and ("criterion_negotiation" in current_strategies or "social_convergence" in current_strategies):
        return "method_negotiation"
    if new_types:
        return "switches_or_broadens"
    if current_strategies != previous_strategies:
        return "strategy_shift"
    return "continues_topic"


def _dependency_semantics(event: dict[str, Any]) -> str:
    evidence_turn_ids = event.get("evidence_turn_ids_available")
    has_evidence = isinstance(evidence_turn_ids, list) and any(
        isinstance(turn_id, int) for turn_id in evidence_turn_ids
    )
    stage = event.get("generation_stage")
    if stage in {"adaptive_followup", "iterative_round_robin"}:
        return "evidence_conditioned" if has_evidence else "evidence_missing"
    if stage in {"baseline_battery", "baseline_probe"}:
        return "preplanned_without_answers"
    if has_evidence:
        return "evidence_conditioned"
    if event.get("interaction_mode") == "interactive_discussion":
        return "freeform_unresolved"
    return "provenance_unavailable"


def _transition_label(topical: str, dependency: str) -> str:
    if topical == "opening_probe":
        return topical
    if dependency == "evidence_conditioned":
        if topical == "same_area_new_angle":
            return "adaptive_deepening"
        if topical == "switches_or_broadens":
            return "adaptive_broadening"
        return "adaptive_followup"
    if dependency == "evidence_missing":
        return "adaptive_provenance_missing"
    if dependency == "preplanned_without_answers":
        if topical == "same_area_new_angle":
            return "preplanned_same_area"
        if topical == "switches_or_broadens":
            return "preplanned_broadening"
        return "preplanned_progression"
    return topical


def _tag_counts_by_round(events: list[dict[str, Any]], field: str) -> dict[str, dict[str, int]]:
    by_round: dict[str, Counter[str]] = defaultdict(Counter)
    for event in events:
        round_key = _round_key(event)
        for tag in event.get(field, []):
            tag_id = tag.get("tag") if isinstance(tag, dict) else None
            if tag_id:
                by_round[round_key][tag_id] += 1
    return {round_key: dict(counts) for round_key, counts in sorted(by_round.items())}


def _round_key(event: dict[str, Any]) -> str:
    if event.get("interaction_mode") == "interactive_discussion" and event.get("phase"):
        return str(event.get("phase"))
    round_index = event.get("round_index")
    if round_index is not None:
        return f"round_{round_index}"
    return str(event.get("phase") or "unknown")


def _tag_set(tags: Any) -> set[str]:
    if not isinstance(tags, list):
        return set()
    return {
        tag.get("tag")
        for tag in tags
        if isinstance(tag, dict) and isinstance(tag.get("tag"), str)
    }


def _primary_tag(tags: Any) -> str | None:
    if not isinstance(tags, list) or not tags:
        return None
    first = tags[0]
    if isinstance(first, dict):
        return first.get("tag")
    return None


def _primary_question_type(tags: Any) -> str | None:
    if not isinstance(tags, list):
        return None
    topical = [
        tag
        for tag in tags
        if isinstance(tag, dict)
        and isinstance(tag.get("tag"), str)
        and tag.get("tag") not in _NON_TOPICAL_QUESTION_TYPES
    ]
    if topical:
        strongest = max(
            enumerate(topical),
            key=lambda item: (len(item[1].get("matched_indicators", [])), -item[0]),
        )[1]
        return strongest.get("tag")
    return _primary_tag(tags)


def _topic_signature(tags: Any) -> set[str]:
    primary = _primary_question_type(tags)
    if not primary:
        return set()
    return {_QUESTION_TYPE_FAMILIES.get(primary, primary)}


def _top1(snapshot: dict[str, Any]) -> str | None:
    ranking = snapshot.get("ranking") or []
    return ranking[0] if ranking else None


def _mean(values: Any) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return sum(materialized) / len(materialized)
