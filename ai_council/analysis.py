from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from ai_council.audit import audit_experiment_behavior
from ai_council.extraction import extract_posthoc_interactions
from ai_council.metrics import compute_evolution_metrics, compute_ranking_metrics
from ai_council.rankings import (
    kendall_tau_against_prior,
    ranking_ids,
    score_r_squared_against_prior,
    spearman_rho_against_prior,
)
from ai_council.storage import RunStore, load_jsonl
from ai_council.spend import compute_spend_lineage
from ai_council.taxonomy import EVALUATION_STRATEGY, QUESTION_TYPE, load_taxonomy, taxonomy_hits_for_entry


def analyze_run(run_dir: str | Path, prior_ranking_file: str | Path | None = None) -> dict[str, Any]:
    run_path = Path(run_dir)
    entries = load_jsonl(run_path / "transcript.jsonl")
    findings = load_jsonl(run_path / "monitor_findings.jsonl")
    revalidation_path = run_path / "revalidation_findings.jsonl"
    revalidation_findings = load_jsonl(revalidation_path) if revalidation_path.exists() else None
    config = _load_optional_json(run_path / "config.json")
    run_summary = _load_optional_json(run_path / "run_summary.json")
    experiment_spend = compute_spend_lineage(run_path)
    extraction = extract_posthoc_interactions(entries)
    behavior_audit = audit_experiment_behavior(entries, extraction, config)

    public_entries = [entry for entry in entries if entry.get("visibility") == "public"]
    private_entries = [entry for entry in entries if entry.get("visibility") == "private"]
    speaker_turns = Counter(entry.get("speaker") for entry in entries)
    phase_turns = Counter(entry.get("phase") for entry in entries)

    criteria = Counter()
    taxonomy = load_taxonomy()
    taxonomy_signals = Counter()
    taxonomy_by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    taxonomy_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    question_types = Counter()
    question_types_by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    question_type_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rankings_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    interaction_streams: dict[str, dict[str, Any]] = {}
    for event in extraction.get("probe_events", []):
        if not isinstance(event, dict):
            continue
        speaker = str(event.get("speaker"))
        for hit in event.get("question_type_tags", []):
            if not isinstance(hit, dict) or not isinstance(hit.get("tag"), str):
                continue
            tag = hit["tag"]
            question_types[tag] += 1
            question_types_by_speaker[speaker][tag] += 1
            if len(question_type_examples[tag]) < 3:
                question_type_examples[tag].append(
                    {
                        "turn_id": event.get("turn_id"),
                        "speaker": event.get("speaker"),
                        "phase": event.get("phase"),
                        "matched_indicators": hit.get("matched_indicators", []),
                        "excerpt": _excerpt(event.get("content", "")),
                    }
                )
    for entry in entries:
        metadata = entry.get("metadata") or {}
        for hit in taxonomy_hits_for_entry(entry, taxonomy):
            dimension = hit.get("dimension", EVALUATION_STRATEGY)
            if dimension == QUESTION_TYPE or not _is_evaluator_strategy_entry(entry):
                continue
            tag = hit["tag"]
            speaker = str(entry.get("speaker"))
            example = {
                "turn_id": entry.get("turn_id"),
                "speaker": entry.get("speaker"),
                "phase": entry.get("phase"),
                "matched_indicators": hit["matched_indicators"],
                "excerpt": _excerpt(entry.get("content", "")),
            }
            taxonomy_signals[tag] += 1
            taxonomy_by_speaker[speaker][tag] += 1
            if len(taxonomy_examples[tag]) < 3:
                taxonomy_examples[tag].append(example)
        stream_id = metadata.get("stream_id")
        if isinstance(stream_id, str):
            stream = interaction_streams.setdefault(
                stream_id,
                {
                    "stream_id": stream_id,
                    "interaction_mode": metadata.get("interaction_mode"),
                    "interviewer": metadata.get("interviewer"),
                    "respondent": metadata.get("respondent"),
                    "turn_ids": [],
                    "role_counts": {},
                    "assessments": [],
                },
            )
            stream["turn_ids"].append(entry.get("turn_id"))
            role = str(metadata.get("interaction_role", "unknown"))
            stream["role_counts"][role] = stream["role_counts"].get(role, 0) + 1

        parsed = entry.get("parsed")
        if not isinstance(parsed, dict):
            continue
        for criterion in _criteria_items(parsed.get("criteria", [])):
            criteria[str(criterion)] += 1
        ranking_field = "ranking" if "ranking" in parsed else "current_ranking" if "current_ranking" in parsed else None
        if ranking_field:
            rankings_by_phase[str(entry.get("phase"))].append(
                {
                    "speaker": entry.get("speaker"),
                    "ranking": parsed.get(ranking_field),
                    "ranking_field": ranking_field,
                    "confidence": parsed.get("confidence"),
                    "round_index": entry.get("round_index"),
                    "scores": parsed.get("scores"),
                    "stream_id": stream_id,
                    "judgment_probe_count": metadata.get("judgment_probe_count"),
                    "judgment_probe_total": metadata.get("judgment_probe_total"),
                    "is_primary_judgment": metadata.get("is_primary_judgment"),
                }
            )
        if isinstance(stream_id, str) and metadata.get("interaction_role") == "assessment":
            interaction_streams[stream_id]["assessments"].append(
                {
                    "turn_id": entry.get("turn_id"),
                    "speaker": entry.get("speaker"),
                    "parsed": parsed,
                }
            )

    summary = {
        "turn_count": len(entries),
        "public_turn_count": len(public_entries),
        "private_turn_count": len(private_entries),
        "speaker_turns": dict(speaker_turns),
        "phase_turns": dict(phase_turns),
        "criteria_frequency": dict(criteria),
        "taxonomy": {
            "name": taxonomy.get("name"),
            "version": taxonomy.get("version"),
            "signal_frequency": dict(taxonomy_signals),
            "signals_by_speaker": {
                speaker: dict(counts)
                for speaker, counts in taxonomy_by_speaker.items()
            },
            "examples": dict(taxonomy_examples),
            "question_type_frequency": dict(question_types),
            "question_types_by_speaker": {
                speaker: dict(counts)
                for speaker, counts in question_types_by_speaker.items()
            },
            "question_type_examples": dict(question_type_examples),
            "probe_strategy_frequency": extraction["summary"].get(
                "strategy_frequency",
                {},
            ),
            "measurement_scope": {
                "question_types": "probe and candidate-question text only",
                "probe_strategies": "probe and candidate-question text only",
                "evaluation_strategies": (
                    "evaluator-authored turns; routed candidate answers excluded; "
                    "structured turns classified from values rather than schema keys"
                ),
            },
        },
        "rankings_by_phase": dict(rankings_by_phase),
        "interaction_streams": interaction_streams,
        "posthoc_extraction": extraction["summary"],
        "behavior_audit": behavior_audit["summary"],
        "monitor_findings": findings,
        "revalidation_findings": revalidation_findings,
        "model_spend": _model_spend(run_summary),
        "experiment_spend": experiment_spend,
    }
    run_metrics = {
        "rankings": compute_ranking_metrics(dict(rankings_by_phase)),
        "evolution": compute_evolution_metrics(extraction),
    }
    summary["metrics"] = _compact_metrics(run_metrics)
    if prior_ranking_file is not None:
        summary["prior_agreement"] = compute_prior_agreement(run_path, rankings_by_phase, prior_ranking_file)
    store = RunStore(run_path)
    store.write_json("posthoc_extraction.json", extraction)
    store.write_json("behavior_audit.json", behavior_audit)
    store.write_json("run_metrics.json", run_metrics)
    store.write_json("analysis_summary.json", summary)
    (run_path / "analysis_report.md").write_text(
        render_analysis_report(run_path, summary, extraction, behavior_audit),
        encoding="utf-8",
    )
    return summary


def render_analysis_report(
    run_path: Path,
    summary: dict[str, Any],
    extraction: dict[str, Any],
    behavior_audit: dict[str, Any] | None = None,
) -> str:
    config = _load_optional_json(run_path / "config.json")
    run_summary = _load_optional_json(run_path / "run_summary.json")
    label_map = _taxonomy_label_map()
    lines = [
        f"# Analysis Report: {config.get('name', run_path.name)}",
        "",
        "## Overview",
        "",
        f"- Turns: {summary['turn_count']} total; {summary['public_turn_count']} public; {summary['private_turn_count']} private.",
        f"- Model calls: {run_summary.get('model_calls', 'unknown')}; reported cost: ${float(run_summary.get('reported_cost_usd', 0.0)):.6f}.",
        f"- Extracted Q/A pairs: {extraction['summary']['qa_pair_count']}; candidate question turns: {extraction['summary']['candidate_question_turn_count']}.",
        f"- Monitor findings: {len(summary.get('monitor_findings', []))}.",
        f"- Behavior-audit findings: {(behavior_audit or {}).get('summary', {}).get('finding_count', 0)}.",
        "",
    ]
    experiment_spend = summary.get("experiment_spend", {})
    has_replay_lineage = (
        isinstance(experiment_spend, dict)
        and experiment_spend.get("run_count", 0) > 1
    )
    if has_replay_lineage:
        completeness_note = (
            ""
            if experiment_spend.get("complete") is True
            else "; minimum recorded because at least one source summary is unavailable"
        )
        lines.insert(
            len(lines) - 1,
            "- Experiment lineage: "
            f"{experiment_spend.get('model_calls', 0)} calls; "
            f"${float(experiment_spend.get('reported_cost_usd', 0.0)):.6f} "
            f"across {experiment_spend.get('run_count')} checkpoints"
            f"{completeness_note}.",
        )
    lines.extend(_participant_section(config))
    spend_for_report = (
        experiment_spend.get("model_spend", {})
        if has_replay_lineage
        else summary.get("model_spend", {})
    )
    lines.extend(
        _model_spend_section(
            spend_for_report,
            title=(
                "Experiment Spend by Model"
                if has_replay_lineage
                else "Reported Spend by Model"
            ),
        )
    )
    lines.extend(
        _top_counts_section(
            "Top Probe Strategies",
            summary["taxonomy"].get("probe_strategy_frequency", {}),
            label_map=label_map,
        )
    )
    lines.extend(_top_counts_section("Top Question Types", summary["taxonomy"].get("question_type_frequency", {}), label_map=label_map))
    lines.extend(_metrics_section(summary))
    lines.extend(_rankings_section(summary))
    lines.extend(_prior_section(summary))
    lines.extend(_monitor_section(summary.get("monitor_findings", [])))
    lines.extend(_behavior_audit_section((behavior_audit or {}).get("findings", [])))
    lines.extend(_qa_highlights_section(extraction))
    return "\n".join(lines).rstrip() + "\n"


def compute_prior_agreement(
    run_path: Path,
    rankings_by_phase: dict[str, list[dict[str, Any]]],
    prior_ranking_file: str | Path,
) -> dict[str, Any]:
    with (run_path / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with Path(prior_ranking_file).open("r", encoding="utf-8") as handle:
        prior_data = json.load(handle)

    model_priors = {
        item["provider_model_id"]: item
        for item in prior_data.get("models", [])
        if "provider_model_id" in item and "estimated_rank" in item
    }
    models = _named_map(config.get("models", {}))
    participant_models = {
        participant["id"]: models[participant["model"]]["model"]
        for participant in config.get("participants", [])
        if participant.get("model") in models
    }
    participant_prior_ranks = {
        participant_id: model_priors[model_id]["estimated_rank"]
        for participant_id, model_id in participant_models.items()
        if model_id in model_priors
    }
    participant_prior_scores = {
        participant_id: model_priors[model_id]["intelligence_score"]
        for participant_id, model_id in participant_models.items()
        if model_id in model_priors
        and isinstance(model_priors[model_id].get("intelligence_score"), (int, float))
    }
    expected_order = sorted(participant_prior_ranks, key=lambda item: participant_prior_ranks[item])

    judgments = []
    for phase, phase_rankings in rankings_by_phase.items():
        for item in phase_rankings:
            ranking = ranking_ids(item.get("ranking"), accept_id_objects=True)
            comparable = [participant_id for participant_id in ranking if participant_id in participant_prior_ranks]
            if len(comparable) < 2:
                continue
            judgments.append(
                {
                    "phase": phase,
                    "speaker": item.get("speaker"),
                    "round_index": item.get("round_index"),
                    "ranking": comparable,
                    "confidence": item.get("confidence"),
                    "kendall_tau": kendall_tau_against_prior(comparable, participant_prior_ranks),
                    "spearman_rho": spearman_rho_against_prior(
                        comparable,
                        participant_prior_ranks,
                    ),
                    "pairwise_accuracy": _pairwise_accuracy(
                        kendall_tau_against_prior(comparable, participant_prior_ranks)
                    ),
                    "score_r_squared": score_r_squared_against_prior(
                        item.get("scores"),
                        participant_prior_scores,
                    ),
                    "top1_matches_prior": comparable[0] == expected_order[0] if expected_order else None,
                    "judgment_probe_count": item.get("judgment_probe_count"),
                    "judgment_probe_total": item.get("judgment_probe_total"),
                    "is_primary_judgment": item.get("is_primary_judgment"),
                }
            )

    return {
        "prior_name": prior_data.get("name"),
        "prior_version": prior_data.get("version"),
        "participant_prior_ranks": participant_prior_ranks,
        "participant_prior_scores": participant_prior_scores,
        "participant_model_ids": participant_models,
        "expected_order": expected_order,
        "judgments": judgments,
    }


def _is_evaluator_strategy_entry(entry: dict[str, Any]) -> bool:
    metadata = entry.get("metadata") or {}
    role = metadata.get("interaction_role")
    if role == "answer":
        return False
    if role is not None:
        return role in {
            "question",
            "discussion",
            "assessment",
            "evidence_card",
            "round_ranking",
            "judge_ranking",
            "probe_comparison",
            "wave_judgment",
            "memory_update",
        }
    return entry.get("visibility") == "public"


def _pairwise_accuracy(tau: float | None) -> float | None:
    return (tau + 1) / 2 if tau is not None else None


def _named_map(items: list[dict[str, Any]] | dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if isinstance(items, dict):
        return items
    return {item["name"]: item for item in items}


def _criteria_items(criteria: Any) -> list[Any]:
    if isinstance(criteria, list):
        return criteria
    if isinstance(criteria, str):
        return [criteria]
    return []


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _participant_section(config: dict[str, Any]) -> list[str]:
    model_items = config.get("models", [])
    models = _named_map(model_items if isinstance(model_items, (list, dict)) else [])
    participants = config.get("participants", [])
    if not isinstance(participants, list) or not participants:
        return []
    lines = ["## Participants", ""]
    for participant in participants:
        if not isinstance(participant, dict):
            continue
        model_ref = participant.get("model")
        model = models.get(model_ref, {})
        lines.append(
            f"- {participant.get('id')}: {model_ref}"
            + (f" (`{model.get('model')}`)" if model.get("model") else "")
        )
    lines.append("")
    judges = config.get("judges", [])
    if isinstance(judges, list) and judges:
        lines.extend(["## Independent Judges", ""])
        for judge in judges:
            if not isinstance(judge, dict):
                continue
            model_ref = judge.get("model")
            model = models.get(model_ref, {})
            lines.append(
                f"- {judge.get('id')}: {model_ref}"
                + (f" (`{model.get('model')}`)" if model.get("model") else "")
            )
        lines.append("")
    return lines


def _model_spend(run_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = run_summary.get("model_spend", {})
    if not isinstance(value, dict):
        return {}
    return {
        str(model_ref): item
        for model_ref, item in value.items()
        if isinstance(item, dict)
    }


def _model_spend_section(
    model_spend: Any,
    *,
    title: str = "Reported Spend by Model",
) -> list[str]:
    if not isinstance(model_spend, dict) or not model_spend:
        return []
    lines = [
        f"## {title}",
        "",
        "| Model | Provider model ID | Calls | Reported cost |",
        "| --- | --- | ---: | ---: |",
    ]
    for model_ref, item in sorted(model_spend.items()):
        if not isinstance(item, dict):
            continue
        cost = item.get("reported_cost_usd", 0.0)
        cost_text = f"${float(cost):.6f}" if isinstance(cost, (int, float)) else "n/a"
        lines.append(
            f"| {model_ref} | `{item.get('provider_model_id', '')}` | "
            f"{item.get('model_calls', 0)} | {cost_text} |"
        )
    lines.extend(["", "Costs are included only when the provider reports them.", ""])
    return lines


def _top_counts_section(
    title: str,
    counts: dict[str, int],
    limit: int = 10,
    *,
    label_map: dict[str, str] | None = None,
) -> list[str]:
    lines = [f"## {title}", ""]
    if not counts:
        lines.extend(["- None detected.", ""])
        return lines
    for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]:
        label = (label_map or {}).get(tag)
        label_text = f"{label} (`{tag}`)" if label and label != tag else tag
        lines.append(f"- {label_text}: {count}")
    lines.append("")
    return lines


def _metrics_section(summary: dict[str, Any]) -> list[str]:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return []
    rankings = metrics.get("rankings")
    evolution = metrics.get("evolution")
    lines = ["## Metrics", ""]
    if isinstance(rankings, dict):
        agreement = rankings.get("final_agreement", {})
        if isinstance(agreement, dict):
            mean_tau = agreement.get("mean_pairwise_tau")
            tau_text = f"{mean_tau:.2f}" if isinstance(mean_tau, (int, float)) else "n/a"
            lines.append(
                "- Final inter-model agreement: "
                f"mean pairwise Kendall tau {tau_text}; "
                f"same top-1 pairs {agreement.get('same_top1_pairs', 0)}/{agreement.get('pair_count', 0)}; "
                f"exact-match pairs {agreement.get('exact_match_pairs', 0)}/{agreement.get('pair_count', 0)}."
            )
        churn = rankings.get("churn_by_speaker", {})
        if isinstance(churn, dict) and churn:
            churn_bits = []
            for speaker, item in sorted(churn.items()):
                if not isinstance(item, dict):
                    continue
                mean_tau = item.get("mean_adjacent_tau")
                tau_text = f"{mean_tau:.2f}" if isinstance(mean_tau, (int, float)) else "n/a"
                churn_bits.append(
                    f"{speaker}: {item.get('top1_changes', 0)} top-1 changes, adjacent tau {tau_text}"
                )
            if churn_bits:
                lines.append("- Ranking churn: " + "; ".join(churn_bits) + ".")
    if isinstance(evolution, dict):
        transitions = evolution.get("transition_counts", {})
        transition_text = _counts_inline(transitions)
        lines.append(
            f"- Probe evolution: {evolution.get('probe_event_count', 0)} probe events"
            + (f"; {transition_text}." if transition_text else ".")
        )
    lines.append("")
    return lines


def _rankings_section(summary: dict[str, Any]) -> list[str]:
    rankings_by_phase = summary.get("rankings_by_phase", {})
    if not rankings_by_phase:
        return ["## Rankings", "", "- No structured rankings parsed.", ""]
    lines = ["## Rankings", ""]
    for phase, rankings in rankings_by_phase.items():
        if not isinstance(rankings, list):
            continue
        lines.append(f"### {phase}")
        lines.append("")
        for item in rankings:
            ranking = ranking_ids(item.get("ranking"), accept_id_objects=True)
            ranking_text = " > ".join(ranking) if ranking else str(item.get("ranking"))
            confidence = item.get("confidence")
            confidence_text = f"; confidence={confidence}" if confidence is not None else ""
            probe_count = item.get("judgment_probe_count")
            probe_text = f"; probes={probe_count}" if probe_count is not None else ""
            lines.append(
                f"- {item.get('speaker')}: {ranking_text}{confidence_text}{probe_text}"
            )
        lines.append("")
    return lines


def _prior_section(summary: dict[str, Any]) -> list[str]:
    prior = summary.get("prior_agreement")
    if not isinstance(prior, dict):
        return []
    lines = ["## Prior Agreement", ""]
    expected_order = prior.get("expected_order", [])
    lines.append(f"- Prior expected order: {' > '.join(expected_order) if expected_order else 'unavailable'}.")
    judgments = prior.get("judgments", [])
    taus = [
        judgment.get("kendall_tau")
        for judgment in judgments
        if isinstance(judgment, dict) and isinstance(judgment.get("kendall_tau"), (int, float))
    ]
    if taus:
        lines.append(f"- Mean Kendall tau across comparable judgments: {sum(taus) / len(taus):.2f}.")
    top1 = [
        judgment.get("top1_matches_prior")
        for judgment in judgments
        if isinstance(judgment, dict) and judgment.get("top1_matches_prior") is not None
    ]
    if top1:
        lines.append(f"- Top-1 prior match rate: {sum(1 for value in top1 if value)}/{len(top1)}.")
    if judgments:
        lines.extend(
            [
                "",
                "| Judge | Round | Probes | Kendall tau | Spearman rho | Pairwise accuracy | Score R2 | Top-1 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for judgment in judgments:
            if not isinstance(judgment, dict):
                continue
            lines.append(
                f"| {judgment.get('speaker')} | {judgment.get('round_index') or ''} | "
                f"{judgment.get('judgment_probe_count') or ''} | "
                f"{_metric_text(judgment.get('kendall_tau'))} | "
                f"{_metric_text(judgment.get('spearman_rho'))} | "
                f"{_metric_text(judgment.get('pairwise_accuracy'))} | "
                f"{_metric_text(judgment.get('score_r_squared'))} | "
                f"{'yes' if judgment.get('top1_matches_prior') else 'no'} |"
            )
    lines.append("")
    return lines


def _monitor_section(findings: list[dict[str, Any]]) -> list[str]:
    lines = ["## Monitor Findings", ""]
    if not findings:
        lines.extend(["- None.", ""])
        return lines
    counts = Counter(str(finding.get("code")) for finding in findings)
    for code, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {code}: {count}")
    lines.append("")
    for finding in findings[:5]:
        evidence = finding.get("evidence")
        evidence_text = f" Evidence: `{_excerpt(evidence, 120)}`" if evidence else ""
        lines.append(
            f"- Turn {finding.get('turn_id')} {finding.get('speaker')}: "
            f"{finding.get('message')}{evidence_text}"
        )
    lines.append("")
    return lines


def _behavior_audit_section(findings: list[dict[str, Any]]) -> list[str]:
    lines = ["## Behavior Audit", ""]
    if not findings:
        lines.extend(["- None.", ""])
        return lines
    counts = Counter(str(finding.get("code")) for finding in findings)
    for code, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {code}: {count}")
    lines.append("")
    for finding in findings[:8]:
        turn = finding.get("turn_id") or finding.get("answer_turn_id")
        actor = finding.get("speaker") or finding.get("respondent_id")
        evidence = finding.get("evidence")
        evidence_text = f" Evidence: `{_excerpt(evidence, 160)}`" if evidence else ""
        lines.append(
            f"- Turn {turn} {actor}: {finding.get('message')}{evidence_text}"
        )
    lines.append("")
    return lines


def _qa_highlights_section(extraction: dict[str, Any]) -> list[str]:
    pairs = extraction.get("qa_pairs", [])
    lines = ["## Extracted Q/A Highlights", ""]
    if not isinstance(pairs, list) or not pairs:
        lines.extend(["- No routed Q/A pairs extracted.", ""])
        return lines
    for pair in pairs[:8]:
        question_tags = ", ".join(tag["label"] for tag in pair.get("question_type_tags", [])[:4])
        tag_text = f" [{question_tags}]" if question_tags else ""
        lines.append(
            f"- Pair {pair.get('pair_id')} {pair.get('interviewer_id')} -> "
            f"{pair.get('respondent_id')}{tag_text}"
        )
        lines.append(f"  - Q: {_excerpt(pair.get('question_text', ''), 180)}")
        lines.append(f"  - A: {_excerpt(pair.get('answer_text', ''), 180)}")
        assessment = pair.get("assessment_parsed")
        if isinstance(assessment, dict) and assessment.get("assessment"):
            lines.append(f"  - Assessment: {_excerpt(assessment.get('assessment'), 180)}")
        elif pair.get("assessment_text"):
            lines.append(f"  - Assessment: {_excerpt(pair.get('assessment_text'), 180)}")
    lines.append("")
    return lines


def _taxonomy_label_map() -> dict[str, str]:
    taxonomy = load_taxonomy()
    items = list(taxonomy.get("tags", [])) + list(taxonomy.get("question_types", []))
    return {
        item["id"]: item.get("label", item["id"])
        for item in items
        if isinstance(item, dict) and "id" in item
    }


def _excerpt(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _compact_metrics(run_metrics: dict[str, Any]) -> dict[str, Any]:
    ranking_metrics = run_metrics.get("rankings", {})
    evolution_metrics = run_metrics.get("evolution", {})
    churn = ranking_metrics.get("churn_by_speaker", {}) if isinstance(ranking_metrics, dict) else {}
    compact_churn = {}
    if isinstance(churn, dict):
        for speaker, item in churn.items():
            if not isinstance(item, dict):
                continue
            compact_churn[speaker] = {
                "snapshot_count": item.get("snapshot_count"),
                "transition_count": item.get("transition_count"),
                "mean_adjacent_tau": item.get("mean_adjacent_tau"),
                "top1_changes": item.get("top1_changes"),
            }
    return {
        "rankings": {
            "snapshot_count": ranking_metrics.get("snapshot_count") if isinstance(ranking_metrics, dict) else 0,
            "final_rankings": ranking_metrics.get("final_rankings", []) if isinstance(ranking_metrics, dict) else [],
            "final_agreement": _compact_agreement(ranking_metrics.get("final_agreement", {}))
            if isinstance(ranking_metrics, dict)
            else {},
            "churn_by_speaker": compact_churn,
        },
        "evolution": {
            "probe_event_count": evolution_metrics.get("probe_event_count", 0)
            if isinstance(evolution_metrics, dict)
            else 0,
            "transition_counts": evolution_metrics.get("transition_counts", {})
            if isinstance(evolution_metrics, dict)
            else {},
            "topical_transition_counts": evolution_metrics.get(
                "topical_transition_counts",
                {},
            )
            if isinstance(evolution_metrics, dict)
            else {},
            "dependency_counts": evolution_metrics.get("dependency_counts", {})
            if isinstance(evolution_metrics, dict)
            else {},
            "question_types_by_round": evolution_metrics.get("question_types_by_round", {})
            if isinstance(evolution_metrics, dict)
            else {},
            "strategies_by_round": evolution_metrics.get("strategies_by_round", {})
            if isinstance(evolution_metrics, dict)
            else {},
            "adaptive": evolution_metrics.get("adaptive", {})
            if isinstance(evolution_metrics, dict)
            else {},
        },
    }


def _compact_agreement(agreement: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(agreement, dict):
        return {}
    return {
        "pair_count": agreement.get("pair_count"),
        "mean_pairwise_tau": agreement.get("mean_pairwise_tau"),
        "same_top1_pairs": agreement.get("same_top1_pairs"),
        "exact_match_pairs": agreement.get("exact_match_pairs"),
        "top1_counts": agreement.get("top1_counts", {}),
        "exact_ranking_counts": agreement.get("exact_ranking_counts", {}),
    }


def _counts_inline(counts: Any, limit: int = 5) -> str:
    if not isinstance(counts, dict) or not counts:
        return ""
    parts = [
        f"{key}={value}"
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]
    return ", ".join(parts)


def _metric_text(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"
